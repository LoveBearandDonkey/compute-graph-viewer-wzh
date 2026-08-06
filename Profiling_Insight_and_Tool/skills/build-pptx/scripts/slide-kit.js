'use strict';

const fs = require('fs');
const path = require('path');

const WIDTH = 1600;
const HEIGHT = 900;

const defaultTheme = Object.freeze({
  bg: '#0B1020',
  surface: '#151E32',
  surfaceAlt: '#1B263D',
  text: '#F5F7FB',
  secondary: '#B8C3D6',
  muted: '#7F8CA5',
  line: '#2B3955',
  accent: '#37C6D0',
  blue: '#4B8DFF',
  green: '#3BC780',
  amber: '#F2B84B',
  red: '#F06476',
  violet: '#9A7BFF',
  font: 'Microsoft YaHei, Segoe UI, Arial, sans-serif',
});

function escapeXml(value) {
  return String(value)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&apos;');
}

function attr(value) {
  return escapeXml(value == null ? '' : value);
}

function color(value, fallback) {
  return value || fallback;
}

function imageMime(filePath) {
  const ext = path.extname(filePath).toLowerCase();
  const mimes = {
    '.png': 'image/png',
    '.jpg': 'image/jpeg',
    '.jpeg': 'image/jpeg',
    '.gif': 'image/gif',
    '.webp': 'image/webp',
    '.svg': 'image/svg+xml',
  };
  if (!mimes[ext]) throw new Error(`Unsupported image type: ${ext}`);
  return mimes[ext];
}

function wrapText(text, maxUnits) {
  const lines = [];
  const charUnits = (ch) => /[\u2e80-\uffff]/.test(ch) ? 1 : 0.55;
  const measure = (value) => [...value].reduce((sum, ch) => sum + charUnits(ch), 0);
  for (const paragraph of String(text).split('\n')) {
    if (!paragraph) {
      lines.push('');
      continue;
    }
    let line = '';
    let used = 0;
    const tokens = paragraph.match(/[^\s\u2e80-\uffff]+|\s+|[\u2e80-\uffff]/g) || [];
    for (const token of tokens) {
      const isSpace = /^\s+$/.test(token);
      const tokenUnits = measure(token);
      if (isSpace && !line) continue;
      if (used + tokenUnits <= maxUnits) {
        line += token;
        used += tokenUnits;
        continue;
      }
      if (line.trim()) lines.push(line.trimEnd());
      line = '';
      used = 0;
      if (isSpace) continue;
      if (tokenUnits <= maxUnits) {
        line = token;
        used = tokenUnits;
      } else {
        for (const ch of token) {
          const units = charUnits(ch);
          if (used + units > maxUnits && line) {
            lines.push(line);
            line = '';
            used = 0;
          }
          line += ch;
          used += units;
        }
      }
    }
    if (line || lines.length === 0) lines.push(line.trimEnd());
  }
  return lines;
}

class Slide {
  constructor(deck, meta = {}) {
    this.deck = deck;
    this.meta = meta;
    this.elements = [];
    this.bounds = [];
    this.defs = [];
    this.background(deck.theme.bg);
  }

  track(type, x, y, w, h) {
    const values = [x, y, w, h];
    if (!values.every(Number.isFinite)) throw new Error(`${type} has non-numeric bounds`);
    if (w < 0 || h < 0) throw new Error(`${type} has negative size`);
    this.bounds.push({ type, x, y, w, h });
  }

  background(fill) {
    this.elements.push(`<rect width="${WIDTH}" height="${HEIGHT}" fill="${attr(fill)}"/>`);
    return this;
  }

  rect(x, y, w, h, options = {}) {
    this.track('rect', x, y, w, h);
    const radius = options.radius == null ? 16 : options.radius;
    const stroke = options.stroke ? ` stroke="${attr(options.stroke)}" stroke-width="${options.strokeWidth || 1}"` : '';
    const opacity = options.opacity == null ? 1 : options.opacity;
    this.elements.push(`<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${radius}" fill="${attr(color(options.fill, this.deck.theme.surface))}" fill-opacity="${opacity}"${stroke}/>`);
    return this;
  }

  ellipse(x, y, w, h, options = {}) {
    this.track('ellipse', x, y, w, h);
    const stroke = options.stroke ? ` stroke="${attr(options.stroke)}" stroke-width="${options.strokeWidth || 1}"` : '';
    this.elements.push(`<ellipse cx="${x + w / 2}" cy="${y + h / 2}" rx="${w / 2}" ry="${h / 2}" fill="${attr(color(options.fill, this.deck.theme.accent))}"${stroke}/>`);
    return this;
  }

  line(x1, y1, x2, y2, options = {}) {
    const left = Math.min(x1, x2);
    const top = Math.min(y1, y2);
    this.track('line', left, top, Math.abs(x2 - x1), Math.abs(y2 - y1));
    const dash = options.dash ? ` stroke-dasharray="${attr(options.dash)}"` : '';
    let marker = '';
    if (options.arrow) {
      const id = `arrow-${this.defs.length + 1}`;
      const strokeColor = color(options.stroke, this.deck.theme.line);
      this.defs.push(`<marker id="${id}" markerWidth="10" markerHeight="10" refX="9" refY="5" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="${attr(strokeColor)}"/></marker>`);
      marker = ` marker-end="url(#${id})"`;
    }
    this.elements.push(`<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${attr(color(options.stroke, this.deck.theme.line))}" stroke-width="${options.strokeWidth || 3}"${dash}${marker}/>`);
    return this;
  }

  text(value, x, y, w, h, options = {}) {
    this.track('text', x, y, w, h);
    const size = options.size || 28;
    const font = options.font || this.deck.theme.font;
    const weight = options.bold ? 700 : (options.weight || 400);
    const anchor = options.align === 'center' ? 'middle' : options.align === 'right' ? 'end' : 'start';
    const textX = options.align === 'center' ? x + w / 2 : options.align === 'right' ? x + w : x;
    const lineHeight = options.lineHeight || Math.round(size * 1.28);
    const maxUnits = Math.max(2, w / (size * 0.92));
    const lines = options.wrap === false ? String(value).split('\n') : wrapText(value, maxUnits);
    const total = lines.length * lineHeight;
    const firstY = options.valign === 'middle'
      ? y + (h - total) / 2 + size
      : options.valign === 'bottom'
        ? y + h - total + size
        : y + size;
    const spans = lines.map((line, index) => `<tspan x="${textX}" dy="${index === 0 ? 0 : lineHeight}">${escapeXml(line)}</tspan>`).join('');
    this.elements.push(`<text x="${textX}" y="${firstY}" fill="${attr(color(options.fill, this.deck.theme.text))}" font-family="${attr(font)}" font-size="${size}" font-weight="${weight}" text-anchor="${anchor}">${spans}</text>`);
    return this;
  }

  image(filePath, x, y, w, h, options = {}) {
    this.track('image', x, y, w, h);
    const resolved = path.resolve(this.deck.baseDir, filePath);
    if (!fs.existsSync(resolved)) throw new Error(`Image not found: ${resolved}`);
    const mime = imageMime(resolved);
    const data = fs.readFileSync(resolved).toString('base64');
    const fit = options.fit === 'cover' ? 'xMidYMid slice' : 'xMidYMid meet';
    const radius = options.radius || 0;
    let clip = '';
    let clipAttr = '';
    if (radius > 0) {
      const clipId = `clip-${this.defs.length + 1}`;
      this.defs.push(`<clipPath id="${clipId}"><rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${radius}"/></clipPath>`);
      clipAttr = ` clip-path="url(#${clipId})"`;
      clip = `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${radius}" fill="none" stroke="${attr(options.stroke || this.deck.theme.line)}" stroke-width="${options.strokeWidth || 1}"/>`;
    }
    this.elements.push(`<image x="${x}" y="${y}" width="${w}" height="${h}" href="data:${mime};base64,${data}" preserveAspectRatio="${fit}"${clipAttr}/>${clip}`);
    return this;
  }

  validate(index) {
    if (this.elements.length <= 1) throw new Error(`Slide ${index} is empty`);
    for (const item of this.bounds) {
      const epsilon = 0.01;
      if (item.x < -epsilon || item.y < -epsilon || item.x + item.w > WIDTH + epsilon || item.y + item.h > HEIGHT + epsilon) {
        throw new Error(`Slide ${index} ${item.type} is outside 1600x900: ${JSON.stringify(item)}`);
      }
    }
  }

  toSvg(index) {
    this.validate(index);
    const defs = this.defs.length ? `<defs>${this.defs.join('')}</defs>` : '';
    return `<?xml version="1.0" encoding="UTF-8"?>\n<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="900" viewBox="0 0 1600 900">${defs}${this.elements.join('')}</svg>\n`;
  }
}

class Deck {
  constructor(options = {}) {
    this.title = options.title || 'Presentation';
    this.author = options.author || 'Codex';
    this.baseDir = path.resolve(options.baseDir || process.cwd());
    this.theme = { ...defaultTheme, ...(options.theme || {}) };
    this.slides = [];
  }

  addSlide(meta, draw) {
    const slide = new Slide(this, typeof meta === 'string' ? { title: meta } : (meta || {}));
    draw(slide, this);
    this.slides.push(slide);
    return slide;
  }
}

function createDeck(options) {
  return new Deck(options);
}

module.exports = { WIDTH, HEIGHT, defaultTheme, createDeck };
