package main

import (
	"bytes"
	"cdrembg/internal/processor"
	"context"
	"errors"
	"io"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"testing"
)

type fakeProcessor func(context.Context, io.ReadSeeker, io.Writer) error

func (f fakeProcessor) Process(c context.Context, r io.ReadSeeker, w io.Writer) error {
	return f(c, r, w)
}
func upload(t *testing.T, names ...string) *http.Request {
	t.Helper()
	var b bytes.Buffer
	form := multipart.NewWriter(&b)
	for _, name := range names {
		part, err := form.CreateFormFile(name, "image.png")
		if err != nil {
			t.Fatal(err)
		}
		_, _ = part.Write([]byte("test"))
	}
	_ = form.Close()
	r := httptest.NewRequest("POST", "/api/remove", &b)
	r.Header.Set("Content-Type", form.FormDataContentType())
	return r
}
func TestRoutes(t *testing.T) {
	h := newHandler(fakeProcessor(func(_ context.Context, _ io.ReadSeeker, w io.Writer) error {
		_, err := w.Write([]byte("PNG"))
		return err
	}))
	for _, test := range []struct {
		method, path string
		status       int
	}{{"GET", "/", 200}, {"GET", "/health", 200}, {"GET", "/missing", 404}, {"GET", "/api/remove", 405}} {
		w := httptest.NewRecorder()
		h.ServeHTTP(w, httptest.NewRequest(test.method, test.path, nil))
		if w.Code != test.status {
			t.Fatal(test, w.Code)
		}
	}
	w := httptest.NewRecorder()
	h.ServeHTTP(w, upload(t, "file"))
	if w.Code != 200 || w.Header().Get("Content-Type") != "image/png" || w.Body.String() != "PNG" {
		t.Fatal(w)
	}
	for _, names := range [][]string{nil, {"wrong"}, {"file", "file"}, {"file", "extra"}} {
		w := httptest.NewRecorder()
		h.ServeHTTP(w, upload(t, names...))
		if w.Code != 400 {
			t.Fatal(names, w.Code)
		}
	}
}
func TestProcessingErrors(t *testing.T) {
	for _, test := range []struct {
		err    error
		status int
	}{{processor.ErrImage, 422}, {errors.New("private native error"), 500}} {
		h := newHandler(fakeProcessor(func(context.Context, io.ReadSeeker, io.Writer) error { return test.err }))
		w := httptest.NewRecorder()
		h.ServeHTTP(w, upload(t, "file"))
		if w.Code != test.status || bytes.Contains(w.Body.Bytes(), []byte("private native")) {
			t.Fatal(w)
		}
	}
}
func TestBusyAdmissionAndRecovery(t *testing.T) {
	entered, release, done := make(chan struct{}), make(chan struct{}), make(chan struct{})
	h := newHandler(fakeProcessor(func(context.Context, io.ReadSeeker, io.Writer) error { close(entered); <-release; return nil }))
	request := upload(t, "file")
	go func() { defer close(done); h.ServeHTTP(httptest.NewRecorder(), request) }()
	<-entered
	w := httptest.NewRecorder()
	h.ServeHTTP(w, upload(t, "file"))
	if w.Code != 503 || w.Header().Get("Retry-After") == "" {
		t.Error(w)
	}
	health := httptest.NewRecorder()
	h.ServeHTTP(health, httptest.NewRequest("GET", "/health", nil))
	if health.Code != 200 {
		t.Error(health)
	}
	close(release)
	<-done
	// A malformed request is admitted after the previous request released its slot.
	w = httptest.NewRecorder()
	h.ServeHTTP(w, upload(t))
	if w.Code != 400 {
		t.Fatal(w)
	}
}
func TestOversizedRequest(t *testing.T) {
	h := newHandler(fakeProcessor(func(context.Context, io.ReadSeeker, io.Writer) error { t.Fatal("processor called"); return nil }))
	var b bytes.Buffer
	form := multipart.NewWriter(&b)
	part, _ := form.CreateFormFile("file", "large.png")
	_, _ = part.Write(make([]byte, maxUpload+128*1024))
	_ = form.Close()
	r := httptest.NewRequest("POST", "/api/remove", &b)
	r.Header.Set("Content-Type", form.FormDataContentType())
	w := httptest.NewRecorder()
	h.ServeHTTP(w, r)
	if w.Code != 413 {
		t.Fatal(w.Code, w.Body.String())
	}
}
