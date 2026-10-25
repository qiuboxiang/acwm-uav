FROM texlive/texlive:TL2025-historic

WORKDIR /work/paper
ENTRYPOINT ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error"]
CMD ["main.tex"]
