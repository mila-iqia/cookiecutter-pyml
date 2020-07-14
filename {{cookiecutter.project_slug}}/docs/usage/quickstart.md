### How to start the project

Put your project instructions here, like

`pip install -e .`


### Add some math to your docs! 

Everybody loves Schrodinger's equation, why not put it everywhere?
```eval_rst
:math:`i \hbar \frac{\partial}{\partial t}\Psi(\mathbf{r},t) = \hat H \Psi(\mathbf{r},t)`
```

You can also add math directly in your docstrings! For an example, click at the docstrings here:
```eval_rst
:py:meth:`{{cookiecutter.project_slug}}.models.model_loader.load_loss`
```

You can even reference them directly anywhere for convenience, because clicking is for the lazy:
```eval_rst
.. autoclass:: {{cookiecutter.project_slug}}.models.model_loader.load_loss
    :show-inheritance:
    :noindex:
```

### More documentation magic

A lot more information about what you can do with these docs is available [here](https://recommonmark.readthedocs.io/en/stable/auto_structify.html)

For example:

``` important:: We can have notes in markdown!
```

More craziness, wow, maybe rst is worth learning!


``` sidebar:: Line numbers and highlights

     emphasis-lines:
       highlights the lines.
     linenos:
       shows the line numbers as well.
     caption:
       shown at the top of the code block.
     name:
       may be referenced with `:ref:` later.
```

``` code-block::
     :linenos:
     :emphasize-lines: 3,5
     :caption: An example code-block with everything turned on.
     :name: Full code-block example

     # Comment line
     import System
     System.run_emphasis_line
     # Long lines in code blocks create a auto horizontal scrollbar
     System.exit!
```
