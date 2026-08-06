'use strict';

function buildDeck({ createDeck, theme }) {
  const deck = createDeck({
    title: 'Replace with presentation title',
    author: 'Codex',
    baseDir: __dirname,
    theme: {
      ...theme,
      accent: '#37C6D0',
    },
  });

  deck.addSlide({ title: 'Cover' }, (slide) => {
    slide.rect(970, 0, 630, 900, { fill: deck.theme.surfaceAlt, radius: 0 });
    slide.text('PRESENTATION CATEGORY', 90, 95, 650, 36, { size: 18, bold: true, fill: deck.theme.accent });
    slide.text('Replace with one memorable\ncentral message', 90, 215, 760, 190, { size: 58, bold: true, lineHeight: 72, wrap: false });
    slide.text('Use one sentence to tell the audience why this matters.', 94, 450, 710, 80, { size: 26, fill: deck.theme.secondary });
    slide.rect(1050, 155, 400, 470, { fill: deck.theme.surface, stroke: deck.theme.accent, strokeWidth: 2 });
    slide.text('ONE\nVISUAL', 1050, 300, 400, 150, { size: 48, bold: true, align: 'center', valign: 'middle', fill: deck.theme.accent, wrap: false });
    slide.text('YYYY.MM.DD', 94, 805, 300, 30, { size: 18, fill: deck.theme.muted });
  });

  deck.addSlide({ title: 'Three conclusions' }, (slide) => {
    slide.text('Replace with a conclusion-led title', 70, 55, 1400, 60, { size: 42, bold: true });
    slide.line(70, 135, 1530, 135, { stroke: deck.theme.line, strokeWidth: 2 });
    const cards = [
      ['01', 'First conclusion', 'Explain the decision-relevant implication, not a feature inventory.', deck.theme.green],
      ['02', 'Second conclusion', 'Use a concise proof point, comparison, or real user tension.', deck.theme.amber],
      ['03', 'Third conclusion', 'End with what the audience should do or believe next.', deck.theme.accent],
    ];
    cards.forEach(([number, title, body, accent], index) => {
      const x = 70 + index * 500;
      slide.rect(x, 205, 440, 510, { fill: deck.theme.surface, stroke: deck.theme.line, strokeWidth: 1 });
      slide.rect(x, 205, 8, 510, { fill: accent, radius: 0 });
      slide.ellipse(x + 38, 248, 58, 58, { fill: accent });
      slide.text(number, x + 38, 259, 58, 28, { size: 18, bold: true, align: 'center', fill: deck.theme.bg });
      slide.text(title, x + 38, 345, 355, 55, { size: 30, bold: true });
      slide.text(body, x + 38, 435, 355, 175, { size: 22, lineHeight: 34, fill: deck.theme.secondary });
    });
  });

  deck.addSlide({ title: 'Process' }, (slide) => {
    slide.text('Show a process only when sequence matters', 70, 55, 1400, 60, { size: 42, bold: true });
    slide.text('Each step should change the state or narrow the decision.', 72, 125, 1100, 42, { size: 22, fill: deck.theme.secondary });
    const steps = [
      ['1', 'Frame', 'Audience and decision'],
      ['2', 'Compare', 'Evidence and alternatives'],
      ['3', 'Explain', 'Cause and implication'],
      ['4', 'Decide', 'Action and owner'],
    ];
    steps.forEach(([number, title, body], index) => {
      const x = 80 + index * 380;
      if (index < steps.length - 1) slide.line(x + 260, 440, x + 370, 440, { stroke: deck.theme.line, strokeWidth: 4, arrow: true });
      slide.ellipse(x, 370, 140, 140, { fill: index === steps.length - 1 ? deck.theme.green : deck.theme.accent });
      slide.text(number, x, 400, 140, 55, { size: 38, bold: true, align: 'center', fill: deck.theme.bg });
      slide.text(title, x - 20, 555, 180, 45, { size: 28, bold: true, align: 'center' });
      slide.text(body, x - 40, 620, 220, 70, { size: 20, align: 'center', fill: deck.theme.secondary });
    });
  });

  deck.addSlide({ title: 'Summary' }, (slide) => {
    slide.text('End with the ideas worth repeating', 70, 55, 1400, 60, { size: 42, bold: true });
    slide.rect(70, 190, 1460, 520, { fill: deck.theme.surface, stroke: deck.theme.line, strokeWidth: 1 });
    const points = [
      ['01', 'One business outcome'],
      ['02', 'One differentiated mechanism'],
      ['03', 'One proof or boundary'],
      ['04', 'One explicit next step'],
    ];
    points.forEach(([number, text], index) => {
      const y = 245 + index * 105;
      slide.text(number, 130, y, 90, 42, { size: 22, bold: true, fill: deck.theme.accent });
      slide.text(text, 240, y - 4, 1100, 52, { size: 31, bold: true });
      if (index < points.length - 1) slide.line(130, y + 70, 1450, y + 70, { stroke: deck.theme.line, strokeWidth: 1 });
    });
    slide.text('Replace every sample phrase before delivery.', 70, 800, 900, 32, { size: 18, fill: deck.theme.muted });
  });

  return deck;
}

module.exports = { buildDeck };
