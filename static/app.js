'use strict';

const $ = (id) => document.getElementById(id);
const MAX_UPLOAD_BYTES = 10 * 1024 * 1024;
const MAX_AUTO_RETRIES = 2;

let busy = false;
let lastFile = null;
let originalURL = null;
let resultURL = null;

function showError(message) {
  $('error').textContent = message;
  $('error').hidden = false;
}

function sleep(ms, signal) {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(resolve, ms);
    signal.addEventListener(
      'abort',
      () => {
        clearTimeout(timer);
        reject(signal.reason || new DOMException('Aborted', 'AbortError'));
      },
      { once: true }
    );
  });
}

$('choose').addEventListener('click', (event) => {
  event.stopPropagation();
  if (!busy) $('file').click();
});

$('drop').addEventListener('click', (event) => {
  if (!busy && !event.target.closest('button')) {
    $('file').click();
  }
});

$('file').addEventListener('change', () => {
  if ($('file').files.length) {
    processFile($('file').files[0]);
  }
  $('file').value = '';
});

$('retry').addEventListener('click', () => {
  if (!busy && lastFile) {
    processFile(lastFile);
  }
});

for (const type of ['dragenter', 'dragover']) {
  $('drop').addEventListener(type, (event) => {
    event.preventDefault();
    if (!busy) $('drop').classList.add('drag');
  });
}

for (const type of ['dragleave', 'drop']) {
  $('drop').addEventListener(type, (event) => {
    event.preventDefault();
    $('drop').classList.remove('drag');
  });
}

$('drop').addEventListener('drop', (event) => {
  if (busy) return;
  const files = event.dataTransfer?.files;
  if (!files || files.length !== 1) {
    showError('Drop one image at a time.');
    return;
  }
  processFile(files[0]);
});

window.addEventListener('dragover', (event) => event.preventDefault());
window.addEventListener('drop', (event) => event.preventDefault());

window.addEventListener('paste', (event) => {
  if (busy) return;
  const items = event.clipboardData?.items;
  if (!items) return;
  for (const item of items) {
    if (item.kind === 'file' && (item.type === 'image/png' || item.type === 'image/jpeg')) {
      const pasted = item.getAsFile();
      if (pasted) {
        event.preventDefault();
        const ext = item.type === 'image/png' ? 'png' : 'jpg';
        const named = pasted.name ? pasted : new File([pasted], `pasted-image.${ext}`, { type: item.type });
        processFile(named);
        return;
      }
    }
  }
});

async function sendWithRetry(file, signal) {
  for (let attempt = 0; ; attempt++) {
    const data = new FormData();
    data.append('file', file);
    const response = await fetch('/api/remove', {
      method: 'POST',
      body: data,
      signal,
    });

    if (response.status === 503 && attempt < MAX_AUTO_RETRIES) {
      const retryHeader = Number.parseInt(response.headers.get('retry-after') || '3', 10);
      const waitSec = Number.isFinite(retryHeader) ? Math.min(10, Math.max(1, retryHeader)) : 3;
      $('status').textContent = `Server is busy processing another image. Retrying in ${waitSec}s (${attempt + 1}/${MAX_AUTO_RETRIES})…`;
      await sleep(waitSec * 1000, signal);
      $('status').textContent = 'Processing your image…';
      continue;
    }
    return response;
  }
}

async function processFile(file) {
  if (busy) return;
  $('error').hidden = true;
  $('retry').hidden = true;

  if (!['image/jpeg', 'image/png'].includes(file.type)) {
    showError('Please choose a JPEG or PNG image.');
    return;
  }
  if (!file.size || file.size > MAX_UPLOAD_BYTES) {
    showError('Choose a non-empty image up to 10 MiB.');
    return;
  }

  lastFile = file;
  busy = true;
  $('choose').disabled = true;
  $('choose').textContent = 'Processing…';
  $('drop').classList.add('busy');
  $('result').setAttribute('aria-busy', 'true');

  if (originalURL) URL.revokeObjectURL(originalURL);
  if (resultURL) URL.revokeObjectURL(resultURL);
  resultURL = null;

  $('after').hidden = true;
  $('after').removeAttribute('src');
  $('download').hidden = true;
  $('download').removeAttribute('href');
  $('dimensions').textContent = 'PNG';
  $('afterEmpty').hidden = false;
  $('afterEmpty').textContent = 'Removing the background…';

  originalURL = URL.createObjectURL(file);
  $('before').src = originalURL;
  $('before').hidden = false;
  $('beforeEmpty').hidden = true;
  $('filename').textContent = file.name;
  $('status').textContent = 'Processing your image. CPU segmentation takes a few seconds.';

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 170000);

  try {
    const response = await sendWithRetry(file, controller.signal);
    if (!response.ok) {
      let message = 'Processing failed. Please try again.';
      try {
        const payload = await response.json();
        if (payload?.error) message = payload.error;
      } catch {
        // Keep fallback message when body is not JSON.
      }
      throw new Error(message);
    }
    if (!response.headers.get('content-type')?.startsWith('image/png')) {
      throw new Error('The server returned an unexpected response format.');
    }

    const processMs = Number.parseInt(response.headers.get('x-process-time-ms') || '', 10);
    const blob = await response.blob();
    resultURL = URL.createObjectURL(blob);

    $('after').src = resultURL;
    try {
      await $('after').decode();
    } catch {
      // Ignore decode hint failure if browser already decoded image.
    }
    $('after').hidden = false;
    $('afterEmpty').hidden = true;

    const w = $('after').naturalWidth;
    const h = $('after').naturalHeight;
    if (w && h) {
      $('dimensions').textContent = `${w}×${h} PNG`;
    }

    const stem = file.name.replace(/\.[^.]*$/, '') || 'image';
    $('download').href = resultURL;
    $('download').download = `${stem}-no-background.png`;
    $('download').hidden = false;

    const timing = Number.isFinite(processMs) ? ` in ${(processMs / 1000).toFixed(2)}s` : '';
    $('status').textContent = `Done${timing}. Your transparent PNG is ready.`;
  } catch (error) {
    const message =
      error?.name === 'AbortError'
        ? 'Processing timed out. The server may still be finishing; retry shortly.'
        : error?.message || 'Processing failed.';
    showError(message);
    $('retry').hidden = false;
    $('status').textContent = 'Click Retry to try again, or choose another image.';
    $('afterEmpty').textContent = 'No result yet';
  } finally {
    clearTimeout(timeout);
    busy = false;
    $('choose').disabled = false;
    $('choose').textContent = 'Choose another image';
    $('drop').classList.remove('busy');
    $('result').setAttribute('aria-busy', 'false');
  }
}
