/* =====================================================================
 * render.js  ·  Config-driven renderer for the research project page.
 *
 * This file is the UI layer. It contains NO paper-specific content:
 * every piece of text, link, figure, and table is read at runtime from
 * config.json (the data layer). To change the page, edit config.json —
 * you should rarely need to touch this file.
 *
 * Supported section/block types (see config.schema.json for the full
 * contract): text, list, figure, table, callout, subtitle, html, video,
 * group. The `slides` and `poster` objects render as embedded PDFs.
 * ===================================================================== */
(function () {
  "use strict";

  var app = document.getElementById("app");
  var CONFIG_PATH = (app && app.getAttribute("data-config")) || "config.json";

  /* ---------- tiny DOM helper ---------- */
  function h(tag, attrs, children) {
    var e = document.createElement(tag);
    if (attrs) {
      for (var k in attrs) {
        if (!Object.prototype.hasOwnProperty.call(attrs, k)) continue;
        var v = attrs[k];
        if (v == null) continue;
        if (k === "class") e.className = v;
        else if (k === "html") e.innerHTML = v;
        else if (k === "text") e.textContent = v;
        else e.setAttribute(k, v);
      }
    }
    appendChildren(e, children);
    return e;
  }

  function appendChildren(e, children) {
    if (children == null) return;
    if (!Array.isArray(children)) children = [children];
    children.forEach(function (c) {
      if (c == null) return;
      e.appendChild(typeof c === "string" ? document.createTextNode(c) : c);
    });
  }

  function alignClass(a) {
    if (a === "left") return "has-text-left";
    if (a === "center") return "has-text-centered";
    return "has-text-justified";
  }


  /* ---------- <head> / SEO / citation meta ---------- */
  function setMetaTag(attr, name, content) {
    if (content == null || content === "") return;
    var sel = "meta[" + attr + '="' + name + '"]';
    var m = document.head.querySelector(sel);
    if (!m) {
      m = document.createElement("meta");
      m.setAttribute(attr, name);
      document.head.appendChild(m);
    }
    m.setAttribute("content", content);
  }

  function addMetaTag(attr, name, content) {
    if (content == null || content === "") return;
    var m = document.createElement("meta");
    m.setAttribute(attr, name);
    m.setAttribute("content", content);
    document.head.appendChild(m);
  }

  function setFavicon(href) {
    var l = document.head.querySelector('link[rel="icon"]');
    if (!l) {
      l = document.createElement("link");
      l.setAttribute("rel", "icon");
      document.head.appendChild(l);
    }
    l.setAttribute("href", href);
  }

  function applyHead(cfg) {
    var site = cfg.site || {};
    if (site.title) document.title = site.title;
    if (site.lang) document.documentElement.setAttribute("lang", site.lang);
    setMetaTag("name", "description", site.description);
    setMetaTag("name", "keywords", site.keywords);
    if (site.favicon) setFavicon(site.favicon);
    // Google Scholar / citation meta
    setMetaTag("name", "citation_title", site.title);
    (cfg.authors || []).forEach(function (a) {
      addMetaTag("name", "citation_author", a.name);
    });
    if (site.publicationDate) setMetaTag("name", "citation_publication_date", site.publicationDate);
    if (site.venue) setMetaTag("name", "citation_conference_title", site.venue);
  }

  /* ---------- link attributes: new tab, or direct download ----------
   * A link that carries `download` points at a file shipped with the page
   * (e.g. the slides PDF): the browser saves it instead of navigating away.
   * `download: "Name.pdf"` also sets the file name it is saved under.
   */
  function linkAttrs(l, cls) {
    var attrs = { href: l.url, "class": cls };
    if (l.download) {
      attrs.download = (typeof l.download === "string") ? l.download : "";
    } else {
      attrs.target = "_blank";
      attrs.rel = "noopener";
    }
    return attrs;
  }

  /* ---------- hero ---------- */
  function buildHero(cfg) {
    var site = cfg.site || {};
    var col = h("div", { "class": "column has-text-centered" });

    col.appendChild(h("h1", { "class": "title is-2 publication-title", html: site.title || "" }));
    if (site.venue) col.appendChild(h("h2", { "class": "publication-venue-line", html: site.venue }));

    var affs = cfg.affiliations || [];
    var multiAff = affs.length > 1;

    var authors = cfg.authors || [];
    if (authors.length) {
      var aDiv = h("div", { "class": "is-size-5 publication-authors" });
      authors.forEach(function (a, i) {
        var block = h("span", { "class": "author-block" });
        if (a.url) block.appendChild(h("a", { href: a.url, target: "_blank", rel: "noopener" }, a.name));
        else block.appendChild(document.createTextNode(a.name));
        if (multiAff && a.affiliations && a.affiliations.length)
          block.appendChild(h("sup", { html: a.affiliations.join(",") }));
        if (a.note) block.appendChild(h("sup", { html: a.note }));
        aDiv.appendChild(block);
        if (i < authors.length - 1) aDiv.appendChild(document.createTextNode(", "));
      });
      col.appendChild(aDiv);
    }

    if (affs.length) {
      var afDiv = h("div", { "class": "is-size-5 publication-affiliations" });
      affs.forEach(function (af, i) {
        var span = h("span", { "class": "affiliation-block" });
        if (multiAff) span.appendChild(h("sup", { html: String(af.id) + " " }));
        span.appendChild(document.createTextNode(af.name));
        afDiv.appendChild(span);
        if (i < affs.length - 1) afDiv.appendChild(document.createTextNode("   "));
      });
      col.appendChild(afDiv);
    }

    var links = (cfg.links || []).filter(function (l) { return l.enabled !== false && l.url; });
    if (links.length) {
      var lc = h("div", { "class": "publication-links" });
      links.forEach(function (l) {
        var a = h("a", linkAttrs(l, "external-link button is-normal is-rounded is-dark"), [
          h("span", { "class": "icon" }, h("i", { "class": l.icon || "fas fa-link" })),
          h("span", null, l.label || l.type || "Link")
        ]);
        lc.appendChild(h("span", { "class": "link-block" }, a));
      });
      col.appendChild(lc);
    }

    return h("section", { "class": "hero" },
      h("div", { "class": "hero-body" },
        h("div", { "class": "container is-max-desktop" },
          h("div", { "class": "columns is-centered" }, col))));
  }

  /* ---------- generic section wrapper ---------- */
  function sectionWrap(col, opts) {
    opts = opts || {};
    var attrs = { "class": "section" };
    if (opts.tightTop) attrs.style = "padding-top:0";
    if (opts.id) attrs.id = opts.id;
    return h("section", attrs,
      h("div", { "class": "container is-max-desktop" },
        h("div", { "class": "columns is-centered" }, col)));
  }

  /* ---------- block renderers (append into a column) ---------- */
  function appendBody(col, node) {
    switch (node.type) {
      case "text":
        col.appendChild(h("div", { "class": "content " + alignClass(node.align), html: "<p>" + (node.body || "") + "</p>" }));
        break;
      case "list":
        var listTag = node.ordered === false ? "ul" : "ol";
        var list = h(listTag, { "class": "contrib-list " + alignClass(node.align || "left") });
        (node.items || []).forEach(function (it) { list.appendChild(h("li", { html: it })); });
        col.appendChild(list);
        break;
      case "figure":
        col.appendChild(buildFigure(node));
        break;
      case "table":
        col.appendChild(buildTable(node));
        break;
      case "callout":
        var cls = "callout" + (node.variant ? " callout-" + node.variant : "");
        col.appendChild(h("div", { "class": cls, html: node.body || "" }));
        break;
      case "subtitle":
        col.appendChild(h("h3", { "class": "title is-4 has-text-left", html: node.text || node.title || "" }));
        break;
      case "video":
        (Array.isArray(node.videos) ? node.videos : (node.url ? [{ url: node.url, caption: node.caption }] : []))
          .forEach(function (v) {
            if (!v || v.enabled === false || !v.url) return;
            if (v.title) col.appendChild(h("h3", { "class": "title is-4 has-text-centered", html: v.title }));
            col.appendChild(buildVideoEmbed(v.url));
            if (v.caption) col.appendChild(h("p", { "class": "fig-caption has-text-centered", html: v.caption }));
          });
        break;
      case "html":
        col.appendChild(h("div", { "class": "content " + alignClass(node.align), html: node.body || "" }));
        break;
      case "group":
        (node.blocks || []).forEach(function (b) { if (b && b.enabled !== false) appendBody(col, b); });
        break;
      default:
        if (node.body) col.appendChild(h("div", { "class": "content", html: node.body }));
    }
  }

  function buildSection(section, opts) {
    if (section.enabled === false) return null;
    var col = h("div", { "class": "column is-four-fifths" });
    if (section.title && section.type !== "subtitle")
      col.appendChild(h("h2", { "class": "title is-3 has-text-centered", html: section.title }));
    appendBody(col, section);
    return sectionWrap(col, opts);
  }

  /* ---------- video embed (YouTube / Vimeo / mp4) ---------- */
  function toYouTubeEmbed(url) {
    if (!url) return url;
    var m = url.match(/(?:youtu\.be\/|youtube\.com\/(?:watch\?v=|embed\/|shorts\/|v\/))([A-Za-z0-9_-]{6,})/);
    return m ? "https://www.youtube.com/embed/" + m[1] : url;
  }

  function buildVideoEmbed(url) {
    var wrap = h("div", { "class": "publication-video" });
    if (/\.(mp4|webm|ogg)(\?.*)?$/i.test(url)) {
      var v = h("video", { controls: "", playsinline: "", preload: "metadata" });
      v.appendChild(h("source", { src: url }));
      wrap.appendChild(v);
    } else {
      wrap.appendChild(h("iframe", {
        src: toYouTubeEmbed(url),
        title: "Embedded video",
        frameborder: "0",
        allow: "accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share",
        referrerpolicy: "strict-origin-when-cross-origin",
        allowfullscreen: ""
      }));
    }
    return wrap;
  }

  /* ---------- figure with graceful placeholder ---------- */
  function buildFigure(node) {
    var frag = document.createDocumentFragment();
    var width = node.width || "100%";
    var img = h("img", {
      src: node.image,
      alt: node.alt || "",
      style: "width:" + width + ";display:block;margin:0 auto;"
    });
    img.addEventListener("error", function () {
      var ph = h("div", { "class": "figure-placeholder", style: "max-width:" + width },
        [
          h("div", { "class": "figure-placeholder-icon", html: "&#x1F5BC;" }),
          h("div", { "class": "figure-placeholder-title", text: node.alt || "Figure" }),
          h("div", { "class": "figure-placeholder-note", text: "Missing image: " + node.image })
        ]);
      if (img.parentNode) img.parentNode.replaceChild(ph, img);
    });
    frag.appendChild(img);
    if (node.caption) frag.appendChild(h("p", { "class": "fig-caption", html: node.caption }));
    return frag;
  }

  /* ---------- table ---------- */
  function buildTable(t) {
    var frag = document.createDocumentFragment();
    var wrap = h("div", { "class": "table-wrap" });
    var table = h("table", { "class": "results-table" });
    if (t.caption) table.appendChild(h("caption", { html: t.caption }));

    var cols = t.columns || [];
    var thead = h("thead");
    var trh = h("tr");
    cols.forEach(function (c) {
      trh.appendChild(h("th", { "class": c.align === "left" ? "t-left" : null, html: c.header || "" }));
    });
    thead.appendChild(trh);
    table.appendChild(thead);

    var tbody = h("tbody");
    (t.rows || []).forEach(function (r) {
      var tr = h("tr", { "class": r["class"] || null });
      (r.cells || []).forEach(function (cell, idx) {
        var val, cellClass = null;
        if (cell && typeof cell === "object") { val = cell.v; cellClass = cell["class"] || null; }
        else { val = (cell == null ? "" : String(cell)); }
        var colAlign = (cols[idx] && cols[idx].align === "left") ? "t-left" : null;
        var klass = [colAlign, cellClass].filter(Boolean).join(" ") || null;
        tr.appendChild(h("td", { "class": klass, html: val }));
      });
      tbody.appendChild(tr);
    });
    table.appendChild(tbody);
    wrap.appendChild(table);
    frag.appendChild(wrap);
    if (t.legend) frag.appendChild(h("p", { "class": "table-legend", html: t.legend }));
    return frag;
  }

  /* ---------- abstract ---------- */
  function buildAbstract(cfg) {
    var ab = cfg.abstract;
    if (!ab || !ab.body) return null;
    var col = h("div", { "class": "column is-four-fifths" });
    col.appendChild(h("h2", { "class": "title is-3 has-text-centered", html: ab.title || "Abstract" }));
    col.appendChild(h("div", { "class": "content has-text-justified", html: "<p>" + ab.body + "</p>" }));
    return sectionWrap(col, {});
  }

  /* ---------- embedded PDF sections (slides, poster) ----------
   * `aspect` sizes the responsive box to the PDF's own page shape and
   * accepts "16:9", "1.237:1", a bare "56.25%", or a width/height number.
   */
  function aspectPadding(aspect) {
    var DEFAULT = "56.25%"; // 16:9
    if (typeof aspect === "number" && aspect > 0) return (100 / aspect).toFixed(4) + "%";
    if (typeof aspect === "string") {
      var s = aspect.trim();
      var m = s.match(/^([0-9.]+)\s*[:\/x]\s*([0-9.]+)$/i);
      if (m && parseFloat(m[1]) > 0) return (parseFloat(m[2]) / parseFloat(m[1]) * 100).toFixed(4) + "%";
      if (/^[0-9.]+%$/.test(s)) return s;
      var n = parseFloat(s);
      if (n > 0) return (100 / n).toFixed(4) + "%";
    }
    return DEFAULT;
  }

  /* Page images (`images` / `image`) are the reliable way to show a document
   * in the page: many browsers — most mobile ones — silently render nothing
   * for an inline PDF. Without them the section falls back to a PDF embed.
   */
  function buildGallery(p, title, imgs) {
    var wrap = h("div", { "class": "pdf-gallery" });
    var frame = h("div", { "class": "pdf-gallery-frame", style: "padding-bottom:" + aspectPadding(p.aspect) });
    var single = imgs.length < 2;
    var img = h("img", {
      src: imgs[0],
      alt: title + (single ? "" : " — page 1 of " + imgs.length),
      draggable: "false"
    });

    if (single && p.file) {
      frame.appendChild(h("a", { href: p.file, target: "_blank", rel: "noopener", title: "Open the full PDF" }, img));
    } else {
      frame.appendChild(img);
    }
    wrap.appendChild(frame);
    if (single) return wrap;

    var i = 0;
    var prev = h("button", { type: "button", "aria-label": "Previous page", html: "&#8249;" });
    var next = h("button", { type: "button", "aria-label": "Next page", html: "&#8250;" });
    var counter = h("span", { "class": "pdf-gallery-counter", "aria-live": "polite" });

    function preload(n) { if (imgs[n]) { var x = new Image(); x.src = imgs[n]; } }
    function show(n) {
      i = Math.min(Math.max(n, 0), imgs.length - 1);
      img.src = imgs[i];
      img.alt = title + " — page " + (i + 1) + " of " + imgs.length;
      counter.textContent = (i + 1) + " / " + imgs.length;
      prev.disabled = i === 0;
      next.disabled = i === imgs.length - 1;
      preload(i + 1); preload(i - 1);
    }
    prev.addEventListener("click", function () { show(i - 1); });
    next.addEventListener("click", function () { show(i + 1); });
    wrap.setAttribute("tabindex", "0");
    wrap.addEventListener("keydown", function (e) {
      if (e.key === "ArrowLeft") { show(i - 1); e.preventDefault(); }
      else if (e.key === "ArrowRight") { show(i + 1); e.preventDefault(); }
    });
    show(0);

    wrap.appendChild(h("div", { "class": "pdf-gallery-nav" }, [prev, counter, next]));
    return wrap;
  }

  function buildPdfSection(node, fallbackTitle, opts) {
    var p = node || {};
    if (p.enabled === false) return null;
    var imgs = (Array.isArray(p.images) ? p.images : (p.image ? [p.image] : [])).filter(Boolean);
    if (!imgs.length && !p.file) return null;
    var title = p.title || fallbackTitle;
    var col = h("div", { "class": "column is-four-fifths has-text-centered" });
    if (p.separator !== false) col.appendChild(h("hr"));
    col.appendChild(h("h2", { "class": "title is-3", html: title }));

    if (imgs.length) {
      col.appendChild(buildGallery(p, title, imgs));
    } else {
      var pdf = h("div", { "class": "publication-pdf", style: "padding-bottom:" + aspectPadding(p.aspect) });
      pdf.appendChild(h("iframe", { src: p.file, title: title, frameborder: "0", loading: "lazy" }));
      col.appendChild(pdf);
    }

    var caption = p.caption;
    if (caption == null && p.file) {
      caption = imgs.length
        ? '<a href="' + p.file + '" target="_blank" rel="noopener">Open the full PDF</a>.'
        : 'If the ' + String(title).toLowerCase() + ' does not display, ' +
          '<a href="' + p.file + '" target="_blank" rel="noopener">open it directly</a>.';
    }
    if (caption) col.appendChild(h("p", { "class": "fig-caption has-text-centered", html: caption }));
    return sectionWrap(col, opts || {});
  }

  /* ---------- bibtex ---------- */
  var COPY_SVG =
    '<svg height="100%" viewBox="0 0 36 36" width="100%">' +
    '<path d="M21.9,8.3H11.3c-0.9,0-1.7,.8-1.7,1.7v12.3h1.7V10h10.6V8.3z M24.6,11.8h-9.7c-1,0-1.8,.8-1.8,1.8v12.3' +
    'c0,1,.8,1.8,1.8,1.8h9.7c1,0,1.8-0.8,1.8-1.8V13.5C26.3,12.6,25.5,11.8,24.6,11.8z M24.6,25.9h-9.7V13.5h9.7V25.9z"></path>' +
    '</svg>';

  function showToast(msg) {
    var t = h("div", { "class": "copy-toast", text: msg });
    document.body.appendChild(t);
    setTimeout(function () { if (t.parentNode) t.parentNode.removeChild(t); }, 1500);
  }

  function copyText(text) {
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(text).then(function () { showToast("BibTeX copied to clipboard!"); })
        .catch(function () { showToast("Copy failed — select the text manually."); });
    } else {
      showToast("Copy not supported — select the text manually.");
    }
  }

  function buildBibtex(cfg) {
    var b = cfg.bibtex;
    if (!b || !b.entry || b.enabled === false) return null;
    var col = h("div", { "class": "column is-four-fifths", style: "position:relative" });
    col.appendChild(h("h2", { "class": "title", html: b.title || "BibTeX" }));
    var content = h("div", { "class": "content has-text-justified", style: "position:relative" });
    var pre = h("pre");
    var code = h("code", { id: "bibtexContent" });
    code.textContent = b.entry;
    pre.appendChild(code);
    content.appendChild(pre);
    var copy = h("div", { "class": "copy-icon", title: "Copy to clipboard", html: COPY_SVG });
    copy.addEventListener("click", function () { copyText(b.entry); });
    content.appendChild(copy);
    col.appendChild(content);
    return h("section", { "class": "section", id: "BibTeX" },
      h("div", { "class": "container is-max-desktop" },
        h("div", { "class": "columns is-centered has-text-centered" }, col)));
  }

  /* ---------- footer ---------- */
  function buildFooter(cfg) {
    var f = cfg.footer || {};
    var content = h("div", { "class": "content has-text-centered" });
    var any = false;
    if (f.showLinks) {
      var links = (cfg.links || []).filter(function (l) { return l.enabled !== false && l.url; });
      if (links.length) {
        var p = h("p", { "class": "footer-icons" });
        links.forEach(function (l) {
          var fa = linkAttrs(l, "icon-link");
          fa.title = l.label || l.type;
          p.appendChild(h("a", fa, h("i", { "class": l.icon || "fas fa-link" })));
        });
        content.appendChild(p);
        any = true;
      }
    }
    if (f.text) { content.appendChild(h("p", { "class": "footer-note", html: f.text })); any = true; }
    if (!any) return null;
    return h("footer", { "class": "footer" },
      h("div", { "class": "container" },
        h("div", { "class": "columns is-centered" },
          h("div", { "class": "column is-8" }, content))));
  }

  /* ---------- orchestration ---------- */
  function render(cfg) {
    applyHead(cfg);
    var frag = document.createDocumentFragment();
    frag.appendChild(buildHero(cfg));
    var abs = buildAbstract(cfg);
    if (abs) frag.appendChild(abs);
    (cfg.sections || []).forEach(function (s) {
      var sec = buildSection(s, {});
      if (sec) frag.appendChild(sec);
    });
    var slides = buildPdfSection(cfg.slides, "Slides", { id: "slides" });
    if (slides) frag.appendChild(slides);
    var poster = buildPdfSection(cfg.poster, "Poster", { id: "poster" });
    if (poster) frag.appendChild(poster);
    var bib = buildBibtex(cfg);
    if (bib) frag.appendChild(bib);
    var foot = buildFooter(cfg);
    if (foot) frag.appendChild(foot);
    app.innerHTML = "";
    app.appendChild(frag);
  }

  function showError(err) {
    var isFile = location.protocol === "file:";
    var box = h("section", { "class": "section" },
      h("div", { "class": "container is-max-desktop" },
        h("div", { "class": "notification config-error" }, [
          h("h2", { "class": "title is-4", text: "Could not load config.json" }),
          isFile
            ? h("div", null, [
                h("p", { html: "The page was opened directly from the file system, so the browser blocked loading <code>config.json</code> (CORS on the <code>file://</code> protocol)." }),
                h("p", { html: "Start a tiny local server from the project folder, then reload:" }),
                h("pre", { text: "python -m http.server 8000" }),
                h("p", { html: "and open <a href=\"http://localhost:8000\">http://localhost:8000</a>. Deploying to GitHub Pages works without this step." })
              ])
            : h("p", { text: "Details: " + (err && err.message ? err.message : String(err)) })
        ])));
    app.innerHTML = "";
    app.appendChild(box);
  }

  function init() {
    if (!app) return;
    fetch(CONFIG_PATH, { cache: "no-store" })
      .then(function (r) { if (!r.ok) throw new Error("HTTP " + r.status + " for " + CONFIG_PATH); return r.json(); })
      .then(render)
      .catch(showError);
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
