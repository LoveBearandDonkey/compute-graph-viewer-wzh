#!/usr/bin/env node
'use strict';

const fs = require('fs');
const path = require('path');
const { createDeck, defaultTheme } = require('./slide-kit');

function parseArgs(argv) {
  const args = {};
  for (let i = 0; i < argv.length; i += 1) {
    const token = argv[i];
    if (!token.startsWith('--')) throw new Error(`Unexpected argument: ${token}`);
    const key = token.slice(2);
    const value = argv[i + 1];
    if (!value || value.startsWith('--')) throw new Error(`Missing value for --${key}`);
    args[key] = value;
    i += 1;
  }
  return args;
}

function cleanGeneratedSvgs(outDir) {
  if (!fs.existsSync(outDir)) {
    fs.mkdirSync(outDir, { recursive: true });
    return;
  }
  for (const name of fs.readdirSync(outDir)) {
    if (/^slide-\d+\.svg$/i.test(name) || name === 'deck-manifest.json') {
      fs.rmSync(path.join(outDir, name), { force: true });
    }
  }
}

function main() {
  const args = parseArgs(process.argv.slice(2));
  if (!args.source || !args.out) {
    throw new Error('Usage: node render-deck.js --source <deck.js> --out <slide_svgs>');
  }
  const source = path.resolve(args.source);
  const outDir = path.resolve(args.out);
  if (!fs.existsSync(source)) throw new Error(`Deck source not found: ${source}`);

  delete require.cache[require.resolve(source)];
  const moduleValue = require(source);
  if (!moduleValue || typeof moduleValue.buildDeck !== 'function') {
    throw new Error('Deck source must export buildDeck({ createDeck, theme })');
  }
  const deck = moduleValue.buildDeck({ createDeck, theme: defaultTheme });
  if (!deck || !Array.isArray(deck.slides) || deck.slides.length === 0) {
    throw new Error('buildDeck must return a deck with at least one slide');
  }

  cleanGeneratedSvgs(outDir);
  const digits = Math.max(2, String(deck.slides.length).length);
  const slideFiles = deck.slides.map((slide, index) => {
    const name = `slide-${String(index + 1).padStart(digits, '0')}.svg`;
    fs.writeFileSync(path.join(outDir, name), slide.toSvg(index + 1), 'utf8');
    return { file: name, title: slide.meta.title || '' };
  });
  const manifest = {
    title: deck.title,
    author: deck.author,
    width: 1600,
    height: 900,
    slideCount: slideFiles.length,
    slides: slideFiles,
  };
  fs.writeFileSync(path.join(outDir, 'deck-manifest.json'), `${JSON.stringify(manifest, null, 2)}\n`, 'utf8');
  process.stdout.write(`${JSON.stringify({ output: outDir, slides: slideFiles.length, manifest: path.join(outDir, 'deck-manifest.json') }, null, 2)}\n`);
}

try {
  main();
} catch (error) {
  process.stderr.write(`${error.stack || error.message}\n`);
  process.exitCode = 1;
}

