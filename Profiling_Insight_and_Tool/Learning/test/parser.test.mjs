import test from 'node:test';
import assert from 'node:assert/strict';
import { parseKnowledge } from '../server.mjs';

test('splits markdown into titled study cards', () => {
  const md = `# 第一章\n## 注意力\n这里是一段足够长的注意力知识内容，用来形成第一张卡片。\n\n> 💡 这是一个需要单独记忆的重要结论，而且内容长度也足够。\n## 并行\n这里是并行策略的说明内容，也足够形成一张知识卡片。`;
  const result = parseKnowledge(md);
  assert.ok(result.cards.length >= 2);
  assert.equal(result.cards[0].chapter, '第一章');
  assert.equal(result.cards[0].title, '注意力');
  assert.equal(result.cards.at(-1).title, '并行');
});

test('keeps fenced code with its card body', () => {
  const md = '# 章节\n## 示例\n这是一段足够长的正文内容，用于解释下面代码。\n\n```js\n# code comment, not heading\nconst x = 1;\n```';
  const result = parseKnowledge(md);
  assert.match(result.cards[0].body, /```js/);
  assert.match(result.cards[0].body, /const x = 1/);
  assert.equal(result.cards[0].chapter, '章节');
});

test('card ids remain stable when another section is inserted', () => {
  const original = '# 章节\n## 保留\n这是一段长度足够且不会发生变化的知识内容。';
  const changed = '# 新章节\n新增一段足够长的知识内容。\n' + original;
  const before = parseKnowledge(original).cards[0];
  const after = parseKnowledge(changed).cards.find(card => card.title === '保留');
  assert.equal(before.id, after.id);
});

test('cards from the same section still receive unique progress ids', () => {
  const md = '# 章节\n## 小节\n' + '第一段知识内容足够长，可以生成首张卡片。'.repeat(18) + '\n\n> 💡 **核心结论**：第二张卡片。';
  const cards = parseKnowledge(md).cards;
  assert.ok(cards.length >= 2);
  assert.notEqual(cards[0].id, cards[1].id);
});

test('never separates a fenced block across cards', () => {
  const longText = '这是一段较长的说明。'.repeat(35);
  const md = `# 章节\n## 图解\n${longText}\n\n\`\`\`\nline one\n\n# code, not a heading\nline two\n\`\`\`\n\n${longText}`;
  const cards = parseKnowledge(md).cards;
  for (const card of cards) {
    const fences = card.body.match(/```/g)?.length || 0;
    assert.equal(fences % 2, 0, `unbalanced fence in ${card.title}`);
    assert.equal(card.chapter, '章节');
  }
});
