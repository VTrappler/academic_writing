(TeX-add-style-hook
 "Chapter2"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-class-options
                     '(("subfiles" "../../Main_ManuscritThese.tex")))
   (TeX-run-style-hooks
    "latex2e"
    "/home/victor/acadwriting/Manuscrit/Text/Chapter2/img/example_cdf_pdf"
    "/home/victor/acadwriting/Manuscrit/Text/Chapter2/img/example_normal"
    "subfiles"
    "subfiles10")
   (LaTeX-add-labels
    "chap:inverse_problem"
    "sec:model_space_data_space"
    "sec:notion_prob_theory"
    "def:prob_event"
    "def:cond_proba"
    "def:random_variable"
    "def:expectation"
    "def:image_measure"
    "def:distribution"
    "ex:X_rv"
    "fig:example_pdf_cdf"
    "eq:variance_def"
    "def:joint_marginal_cond_densities"
    "eq:marginals_def"
    "eq:marginal_conditioned"
    "ex:gaussian_distribution"
    "fig:example_normal"
    "ssec:bayes_theorem"
    "ssec:inv_problem"
    "sec:frequentist_inference_MLE"
    "eq:lik_gaussian"
    "def:mle"
    "eq:likelihood_definition"
    "eq:def_MLE"
    "sec:bayesian_inference_MAP"
    "eq:bayes_posterior"
    "sec:choice_prior"
    "sec:bayes_point_estimates"
    "eq:def_MMAE"
    "eq:def_MAP")
   (LaTeX-add-bibliographies
    "/home/victor/acadwriting/bibzotero"))
 :latex)

