const fs = require("node:fs");
const path = require("node:path");
const { chromium } = require("playwright");

const root = path.resolve(__dirname, "..");
const reportMd = fs.readFileSync(path.join(root, "report.md"), "utf8");

function escapeHtml(text) {
  return text
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function inline(text) {
  let out = escapeHtml(text);
  out = out.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");
  out = out.replace(/`([^`]+)`/g, "<code>$1</code>");
  return out;
}

function renderMarkdown(md) {
  const lines = md.split(/\r?\n/);
  const html = [];
  let inCode = false;
  let codeLines = [];
  let listItems = [];
  let tableLines = [];

  function flushList() {
    if (listItems.length) {
      html.push("<ul>" + listItems.map((item) => `<li>${inline(item)}</li>`).join("") + "</ul>");
      listItems = [];
    }
  }

  function flushTable() {
    if (!tableLines.length) return;
    const rows = tableLines
      .filter((line) => !/^\|\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?$/.test(line))
      .map((line) => line.replace(/^\||\|$/g, "").split("|").map((cell) => inline(cell.trim())));
    if (rows.length) {
      const [head, ...body] = rows;
      html.push("<table><thead><tr>" + head.map((c) => `<th>${c}</th>`).join("") + "</tr></thead><tbody>");
      for (const row of body) html.push("<tr>" + row.map((c) => `<td>${c}</td>`).join("") + "</tr>");
      html.push("</tbody></table>");
    }
    tableLines = [];
  }

  for (const line of lines) {
    const trimmed = line.trim();
    if (trimmed.startsWith("```")) {
      flushList();
      flushTable();
      if (inCode) {
        html.push(`<pre><code>${escapeHtml(codeLines.join("\n"))}</code></pre>`);
        codeLines = [];
        inCode = false;
      } else {
        inCode = true;
      }
      continue;
    }
    if (inCode) {
      codeLines.push(line);
      continue;
    }
    if (trimmed.startsWith("|")) {
      flushList();
      tableLines.push(trimmed);
      continue;
    }
    flushTable();
    if (!trimmed) {
      flushList();
      continue;
    }
    const image = trimmed.match(/^!\[(.*?)\]\((.*?)\)$/);
    if (image) {
      flushList();
      html.push(`<figure><img src="${image[2]}" alt="${escapeHtml(image[1])}"></figure>`);
      continue;
    }
    if (trimmed.startsWith("- ")) {
      listItems.push(trimmed.slice(2));
      continue;
    }
    if (/^\d+\. /.test(trimmed)) {
      listItems.push(trimmed.replace(/^\d+\. /, ""));
      continue;
    }
    flushList();
    if (trimmed.startsWith("# ")) html.push(`<h1>${inline(trimmed.slice(2))}</h1>`);
    else if (trimmed.startsWith("## ")) html.push(`<h2>${inline(trimmed.slice(3))}</h2>`);
    else if (trimmed.startsWith("### ")) html.push(`<h3>${inline(trimmed.slice(4))}</h3>`);
    else html.push(`<p>${inline(trimmed)}</p>`);
  }
  flushList();
  flushTable();
  return html.join("\n");
}

const html = `<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<style>
  @page { size: A4; margin: 10mm 11mm; }
  body {
    font-family: "PingFang SC", "Songti SC", "STHeiti", "Microsoft YaHei", sans-serif;
    color: #111827;
    font-size: 10px;
    line-height: 1.42;
  }
  h1 { font-size: 22px; margin: 0 0 10px; color: #0f172a; }
  h2 { font-size: 15px; margin: 12px 0 6px; color: #1f2937; break-after: avoid; }
  h3 { font-size: 12px; margin: 8px 0 4px; color: #334155; break-after: avoid; }
  p { margin: 3px 0 5px; }
  ul { margin: 3px 0 6px 18px; padding: 0; }
  li { margin: 1px 0; }
  code { font-family: Menlo, Consolas, monospace; background: #f1f5f9; padding: 0 2px; border-radius: 2px; }
  pre { background: #f8fafc; border: 1px solid #cbd5e1; border-radius: 4px; padding: 5px 7px; margin: 4px 0 7px; white-space: pre-wrap; font-size: 8px; line-height: 1.25; }
  table { width: 100%; border-collapse: collapse; margin: 5px 0 8px; font-size: 9px; break-inside: avoid; }
  th { background: #1f2937; color: white; text-align: left; }
  th, td { border: 1px solid #cbd5e1; padding: 3px 5px; vertical-align: top; }
  tr:nth-child(even) td { background: #f8fafc; }
  figure { margin: 6px 0 8px; text-align: center; break-inside: avoid; }
  img { max-width: 100%; max-height: 245px; object-fit: contain; }
  h2:nth-of-type(9) { break-before: page; }
  h2:nth-of-type(10) { break-before: page; }
</style>
</head>
<body>${renderMarkdown(reportMd)}</body>
</html>`;

const htmlPath = path.join(root, "tmp", "report.html");
fs.mkdirSync(path.dirname(htmlPath), { recursive: true });
fs.writeFileSync(htmlPath, html, "utf8");

(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage();
  await page.goto(`file://${htmlPath}`, { waitUntil: "networkidle" });
  await page.pdf({
    path: path.join(root, "report.pdf"),
    format: "A4",
    printBackground: true,
    preferCSSPageSize: true,
  });
  await browser.close();
})();
