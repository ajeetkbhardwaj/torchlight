// MathJax configuration for torchlight docs.
// pymdownx.arithmatex (generic: false) already converts $...$ -> \(...\)
// and $$...$$ -> \[...\], so only those two forms are configured here.
window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["$$", "$$"], ["\\[", "\\]"]],
    processEscapes: true,
  },
};