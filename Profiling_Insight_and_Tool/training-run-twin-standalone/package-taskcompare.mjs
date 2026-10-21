import { readFileSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { Script } from 'node:vm';

const folder = resolve(dirname(fileURLToPath(import.meta.url)));
const source = join(folder, 'TaskCompare.html');
const output = join(folder, 'TaskCompare-share.html');
let html = readFileSync(source, 'utf8');

let cssCount = 0;
html = html.replace(/<link rel="stylesheet" href="\.\/css\/([^"/]+)">/g, (_, name) => {
  let css = readFileSync(join(folder, 'css', name), 'utf8');
  // style.css repeats the first three linked stylesheets through imports.
  css = css.replace(/^@import\s+['"]\.\/[^'"]+['"];\s*$/gm, '');
  cssCount++;
  return `<style data-bundled-from="css/${name}">\n${css}\n</style>`;
});
if (cssCount !== 6) throw new Error(`Expected 6 stylesheets, got ${cssCount}`);

let scriptCount = 0;
html = html.replace(/<script src="\.\/js\/([^"/]+)"><\/script>/g, (_, name) => {
  const js = readFileSync(join(folder, 'js', name), 'utf8');
  scriptCount++;
  return `<script data-bundled-from="js/${name}">\n${js}\n</script>`;
});
if (scriptCount !== 2) throw new Error(`Expected 2 scripts, got ${scriptCount}`);

const covers = [...new Set([...html.matchAll(/\bpic:'([^']+\.png)'/g)].map(match => match[1]))];
if (covers.length !== 6) throw new Error(`Expected 6 cover images, got ${covers.length}`);
for (const name of covers) {
  const base64 = readFileSync(join(folder, 'pic', name)).toString('base64');
  html = html.replaceAll(`pic:'${name}'`, `pic:'data:image/png;base64,${base64}'`);
}
html = html.replace('src="./pic/${d.pic}"', 'src="${d.pic}"');

// Keep citation labels as text while removing their destination URLs.
let removedLinks = 0;
html = html.replace(/<a\s+href=\\?"https?:\/\/[^"\\]+\\?"\s+target=\\?"_blank\\?">([\s\S]*?)<\/a>/g,
  (_, label) => { removedLinks++; return label; });
if (removedLinks !== 20) throw new Error(`Expected 20 external anchors, got ${removedLinks}`);

let removedDataLinks = 0;
html = html.replace(/^\s+link: "https?:\/\/[^"\r\n]+",\r?\n/gm,
  () => { removedDataLinks++; return ''; });
if (removedDataLinks !== 2) throw new Error(`Expected 2 diagnostic links, got ${removedDataLinks}`);

for (const id of ['navMonitoring', 'navConfigRelation']) {
  const button = new RegExp(`^[ \\t]*<button[^>]*id="${id}"[^>]*>[\\s\\S]*?<\\/button>\\r?\\n`, 'm');
  if (!button.test(html)) throw new Error(`Missing navigation button: ${id}`);
  html = html.replace(button, '');
}
const navHandlers = /const navMonitoringBtn = document\.getElementById\('navMonitoring'\);[\s\S]*?const navConfigRelationBtn = document\.getElementById\('navConfigRelation'\);[\s\S]*?\}\);\r?\n/;
if (!navHandlers.test(html)) throw new Error('Missing standalone navigation handlers');
html = html.replace(navHandlers, '');

for (const [label, pattern] of Object.entries({
  'remote URL': /https?:\/\//,
  'external anchor': /<a\s+href=/i,
  'external asset': /(?:src|href)="\.\//,
  'CSS import': /@import\s+['"]\.\//,
  'sibling navigation': /window\.location\.href\s*=/,
})) {
  if (pattern.test(html)) throw new Error(`Bundle still contains ${label}`);
}

for (const [, js] of html.matchAll(/<script(?:\s+[^>]*)?>([\s\S]*?)<\/script>/g)) {
  new Script(js);
}

writeFileSync(output, html, 'utf8');
console.log(`Created ${output} (${Buffer.byteLength(html).toLocaleString()} bytes)`);
