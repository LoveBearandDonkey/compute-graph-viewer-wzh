#!/usr/bin/env node
'use strict';

const fs = require('fs');
const path = require('path');
const { spawn } = require('child_process');

function parseArgs(argv) {
  const args = { eval: [] };
  for (let i = 0; i < argv.length; i += 1) {
    const token = argv[i];
    if (!token.startsWith('--')) throw new Error(`Unexpected argument: ${token}`);
    const key = token.slice(2);
    const value = argv[i + 1];
    if (!value || value.startsWith('--')) throw new Error(`Missing value for --${key}`);
    if (key === 'eval') args.eval.push(value); else args[key] = value;
    i += 1;
  }
  return args;
}

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

class Cdp {
  constructor(wsUrl) {
    this.socket = new WebSocket(wsUrl);
    this.sequence = 0;
    this.pending = new Map();
    this.events = new Map();
  }

  async open() {
    await new Promise((resolve, reject) => {
      this.socket.onopen = resolve;
      this.socket.onerror = reject;
      this.socket.onmessage = (event) => {
        const message = JSON.parse(event.data);
        if (message.id) {
          const pending = this.pending.get(message.id);
          if (pending) {
            this.pending.delete(message.id);
            message.error ? pending.reject(message.error) : pending.resolve(message.result);
          }
        } else if (message.method) {
          for (const handler of this.events.get(message.method) || []) handler(message.params);
        }
      };
    });
  }

  call(method, params = {}) {
    const id = ++this.sequence;
    this.socket.send(JSON.stringify({ id, method, params }));
    return new Promise((resolve, reject) => this.pending.set(id, { resolve, reject }));
  }

  waitEvent(method, timeoutMs = 20000) {
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => reject(new Error(`Timed out waiting for ${method}`)), timeoutMs);
      const handler = (params) => {
        clearTimeout(timer);
        this.events.set(method, (this.events.get(method) || []).filter((item) => item !== handler));
        resolve(params);
      };
      this.events.set(method, [...(this.events.get(method) || []), handler]);
    });
  }

  close() {
    this.socket.close();
  }
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (!args.url || !args.output) {
    throw new Error('Usage: node capture-webpage.js --url <http-url> --output <image.png> [--wait 2500] [--eval <javascript>]');
  }
  if (!/^https?:\/\//i.test(args.url)) throw new Error('Use an http:// or https:// URL; serve local pages over HTTP');

  const output = path.resolve(args.output);
  fs.mkdirSync(path.dirname(output), { recursive: true });
  const width = Number(args.width || 1600);
  const height = Number(args.height || 900);
  const waitMs = Number(args.wait || 2000);
  const port = Number(args.port || 9223);
  const edge = args.edge || 'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe';
  if (!fs.existsSync(edge)) throw new Error(`Microsoft Edge not found: ${edge}`);
  const profileDir = path.resolve(args.profile || path.join(path.dirname(output), '.edge-profile'));
  fs.mkdirSync(profileDir, { recursive: true });

  const browser = spawn(edge, [
    '--headless=new', '--disable-gpu', '--no-first-run', '--no-default-browser-check',
    `--remote-debugging-port=${port}`, `--user-data-dir=${profileDir}`,
    `--window-size=${width},${height}`, 'about:blank',
  ], { windowsHide: true, stdio: 'ignore' });

  let cdp;
  try {
    let ready = false;
    for (let attempt = 0; attempt < 40; attempt += 1) {
      try {
        const response = await fetch(`http://127.0.0.1:${port}/json/version`);
        if (response.ok) { ready = true; break; }
      } catch (_) {}
      await sleep(250);
    }
    if (!ready) throw new Error('Edge remote debugging endpoint did not start');

    const targetResponse = await fetch(`http://127.0.0.1:${port}/json/new?${encodeURIComponent(args.url)}`, { method: 'PUT' });
    if (!targetResponse.ok) throw new Error(`Could not create browser target: ${targetResponse.status}`);
    const target = await targetResponse.json();
    cdp = new Cdp(target.webSocketDebuggerUrl);
    await cdp.open();
    await cdp.call('Page.enable');
    await cdp.call('Runtime.enable');
    await cdp.call('Emulation.setDeviceMetricsOverride', { width, height, deviceScaleFactor: 1, mobile: false });
    const loaded = cdp.waitEvent('Page.loadEventFired').catch(() => null);
    await cdp.call('Page.navigate', { url: args.url });
    await loaded;
    await sleep(waitMs);
    for (const expression of args.eval) {
      await cdp.call('Runtime.evaluate', { expression, awaitPromise: true, returnByValue: true });
      await sleep(500);
    }
    const shot = await cdp.call('Page.captureScreenshot', { format: 'png', fromSurface: true, captureBeyondViewport: false });
    fs.writeFileSync(output, Buffer.from(shot.data, 'base64'));
    process.stdout.write(`${JSON.stringify({ output, width, height, bytes: fs.statSync(output).size }, null, 2)}\n`);
  } finally {
    if (cdp) {
      await cdp.call('Page.close').catch(() => null);
      cdp.close();
    }
    if (!browser.killed) browser.kill();
    await sleep(250);
  }
}

main()
  .then(() => process.exit(0))
  .catch((error) => {
    process.stderr.write(`${error.stack || error.message}\n`);
    process.exit(1);
  });
