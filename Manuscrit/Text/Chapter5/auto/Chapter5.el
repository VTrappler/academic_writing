(TeX-add-style-hook
 "Chapter5"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-class-options
                     '(("subfiles" "../../Main_ManuscritThese.tex")))
   (TeX-run-style-hooks
    "latex2e"
    "subfiles"
    "subfiles10")
   (LaTeX-add-labels
    "chap:croco")
   (LaTeX-add-bibliographies
    "../../Bibliography"))
 :latex)

