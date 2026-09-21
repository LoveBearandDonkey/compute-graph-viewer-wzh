#!/usr/bin/env node
/* 把一份 pattern 的**默认 URL 状态**写进页面本身。
   用法：node inject-pattern-defaults.cjs <pattern.html> "view=chain&card=1&vtab=3d"

   为什么要有这一步：/patterns/<id>/ 这几份是同一个 demo.html 的不同**默认取景**，
   区别全在 query 上。若只靠使用者自己补参数，那么「打开链接就是这一屏」这个承诺
   就不成立——一条链接必须自带它那一屏。注入的脚本只在参数**缺席**时才填，显式传
   的一律不覆盖（读者手上那条带参数的链接永远优先）。
   embed=1 一律补上：pattern 是拿去嵌的形态，顶栏页脚都收起来。

   deploy.yml 里原来是把这段 JS 用 node -e 写在 YAML 里的（net-slicing 那一份还是），
   多一层 shell/YAML 引号转义，改一个字符就容易连引号一起改错——抽成文件之后可以
   本地直接跑、直接读。 */
const fs = require('fs');
const [file, query] = process.argv.slice(2);
if (!file || !query) {
  console.error('用法: node inject-pattern-defaults.cjs <pattern.html> "k=v&k2=v2"');
  process.exit(2);
}
const pairs = query.split('&').filter(Boolean).map(function (s) {
  const i = s.indexOf('=');
  return [s.slice(0, i), s.slice(i + 1)];
});
if (!pairs.some(function (p) { return p[0] === 'embed'; })) pairs.push(['embed', '1']);

const sets = pairs.map(function (p) {
  return "if(!q.has('" + p[0] + "'))q.set('" + p[0] + "','" + p[1] + "');";
}).join('');
const inject = '<script>(function(){var q=new URLSearchParams(location.search);'
  + sets
  + "var t=location.pathname+'?'+q.toString()+location.hash;"
  + "if(t!==location.pathname+location.search+location.hash)history.replaceState(null,'',t);"
  + '})();</script>';

let html = fs.readFileSync(file, 'utf8');
if (html.indexOf('</head>') < 0) {
  console.error('❌ ' + file + ' 里没有 </head>，注入点不存在');
  process.exit(1);
}
html = html.replace('</head>', inject + '\n</head>');
fs.writeFileSync(file, html);
console.log('injected defaults into ' + file + ': ' + pairs.map(function (p) { return p.join('='); }).join(' · '));
