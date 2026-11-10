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
	"strings"
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
	r := httptest.NewRequest(http.MethodPost, "/api/remove", &b)
	r.Header.Set("Content-Type", form.FormDataContentType())
	return r
}

func TestRoutesAndSecurityHeaders(t *testing.T) {
	h := newHandler(fakeProcessor(func(_ context.Context, _ io.ReadSeeker, w io.Writer) error {
		_, err := w.Write([]byte("PNG"))
		return err
	}))

	cases := []struct {
		method string
		path   string
		status int
		ctype  string
	}{
		{http.MethodGet, "/", http.StatusOK, "text/html"},
		{http.MethodGet, "/styles.css", http.StatusOK, "text/css"},
		{http.MethodGet, "/app.js", http.StatusOK, "text/javascript"},
		{http.MethodGet, "/health", http.StatusOK, "application/json"},
		{http.MethodGet, "/missing", http.StatusNotFound, ""},
		{http.MethodGet, "/api/remove", http.StatusMethodNotAllowed, ""},
	}
	for _, tc := range cases {
		w := httptest.NewRecorder()
		h.ServeHTTP(w, httptest.NewRequest(tc.method, tc.path, nil))
		if w.Code != tc.status {
			t.Fatalf("%s %s: got %d, want %d", tc.method, tc.path, w.Code, tc.status)
		}
		if tc.ctype != "" && !strings.HasPrefix(w.Header().Get("Content-Type"), tc.ctype) {
			t.Fatalf("%s %s: unexpected Content-Type %q", tc.method, tc.path, w.Header().Get("Content-Type"))
		}
		if csp := w.Header().Get("Content-Security-Policy"); !strings.Contains(csp, "default-src 'none'") || strings.Contains(csp, "unsafe-inline") {
			t.Fatalf("expected strict CSP without unsafe-inline, got %q", csp)
		}
	}

	w := httptest.NewRecorder()
	h.ServeHTTP(w, upload(t, "file"))
	if w.Code != http.StatusOK || w.Header().Get("Content-Type") != "image/png" || w.Body.String() != "PNG" || w.Header().Get("X-Process-Time-Ms") == "" {
		t.Fatalf("unexpected success response: code=%d headers=%v body=%q", w.Code, w.Header(), w.Body.String())
	}

	for _, names := range [][]string{nil, {"wrong"}, {"file", "file"}, {"file", "extra"}} {
		w := httptest.NewRecorder()
		h.ServeHTTP(w, upload(t, names...))
		if w.Code != http.StatusBadRequest {
			t.Fatalf("names=%v: got %d, want 400", names, w.Code)
		}
	}
}

func TestProcessingErrors(t *testing.T) {
	cases := []struct {
		err    error
		status int
	}{
		{processor.ErrUnsupportedFormat, http.StatusUnprocessableEntity},
		{processor.ErrDimensionsExceeded, http.StatusUnprocessableEntity},
		{errors.New("private native error"), http.StatusInternalServerError},
	}
	for _, tc := range cases {
		h := newHandler(fakeProcessor(func(context.Context, io.ReadSeeker, io.Writer) error { return tc.err }))
		w := httptest.NewRecorder()
		h.ServeHTTP(w, upload(t, "file"))
		if w.Code != tc.status || bytes.Contains(w.Body.Bytes(), []byte("private native")) {
			t.Fatalf("got status=%d body=%q", w.Code, w.Body.String())
		}
	}
}

func TestBusyAdmissionAndFastHeaderValidation(t *testing.T) {
	entered, release, done := make(chan struct{}), make(chan struct{}), make(chan struct{})
	h := newHandler(fakeProcessor(func(context.Context, io.ReadSeeker, io.Writer) error {
		close(entered)
		<-release
		return nil
	}))

	go func() {
		defer close(done)
		h.ServeHTTP(httptest.NewRecorder(), upload(t, "file"))
	}()
	<-entered

	// Invalid Content-Type fails immediately with 400 without contending for the inference slot.
	badReq := httptest.NewRequest(http.MethodPost, "/api/remove", strings.NewReader("plain text"))
	badReq.Header.Set("Content-Type", "text/plain")
	badRes := httptest.NewRecorder()
	h.ServeHTTP(badRes, badReq)
	if badRes.Code != http.StatusBadRequest {
		t.Fatalf("expected 400 for invalid Content-Type during busy state, got %d", badRes.Code)
	}

	// Valid concurrent upload receives 503 with Retry-After.
	busyRes := httptest.NewRecorder()
	h.ServeHTTP(busyRes, upload(t, "file"))
	if busyRes.Code != http.StatusServiceUnavailable || busyRes.Header().Get("Retry-After") == "" {
		t.Fatalf("expected 503 with Retry-After, got %d %v", busyRes.Code, busyRes.Header())
	}

	health := httptest.NewRecorder()
	h.ServeHTTP(health, httptest.NewRequest(http.MethodGet, "/health", nil))
	if health.Code != http.StatusOK {
		t.Fatalf("expected 200 health while busy, got %d", health.Code)
	}

	close(release)
	<-done

	// Slot recovers cleanly after active request finishes.
	recovered := httptest.NewRecorder()
	h.ServeHTTP(recovered, upload(t))
	if recovered.Code != http.StatusBadRequest {
		t.Fatalf("expected 400 after slot recovery, got %d", recovered.Code)
	}
}

func TestOversizedRequest(t *testing.T) {
	h := newHandler(fakeProcessor(func(context.Context, io.ReadSeeker, io.Writer) error {
		t.Fatal("processor called on oversized upload")
		return nil
	}))
	var b bytes.Buffer
	form := multipart.NewWriter(&b)
	part, _ := form.CreateFormFile("file", "large.png")
	_, _ = part.Write(make([]byte, maxUpload+128*1024))
	_ = form.Close()

	r := httptest.NewRequest(http.MethodPost, "/api/remove", &b)
	r.Header.Set("Content-Type", form.FormDataContentType())
	w := httptest.NewRecorder()
	h.ServeHTTP(w, r)
	if w.Code != http.StatusRequestEntityTooLarge {
		t.Fatalf("expected 413, got %d %s", w.Code, w.Body.String())
	}
}

func TestHealthCheckHelper(t *testing.T) {
	h := newHandler(fakeProcessor(func(context.Context, io.ReadSeeker, io.Writer) error { return nil }))
	srv := httptest.NewServer(h)
	defer srv.Close()

	if err := checkHealth(srv.URL + "/health"); err != nil {
		t.Fatalf("expected healthy check, got %v", err)
	}
	if err := checkHealth(srv.URL + "/missing"); err == nil {
		t.Fatal("expected error for non-200 status")
	}
}
