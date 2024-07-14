Installation
############

At the moment, PyJama can only be installed from source, but a pip package is planned for the future.

In either case, we recommend you install this package within a virtual environment.
Two of the most popular tools for creating virtual environments are `conda <https://docs.conda.io>`_ and `venv <https://docs.python.org/3/library/venv.html>`_.

Installation from source
------------------------
1.) Clone the repository into a directory of your choice and navigate into it:

.. code-block:: bash

    git clone git@github.com:huoxiaoyao/pyjama.git && cd pyjama

2.) Install the package (make sure you have your virtual environment activated before running this command):

.. code-block:: bash

    make install

3.) Test the installation in Python:

.. code-block:: bash

    python

.. code-block:: python

    >>> import pyjama
    >>> print(pyjama.__version__)
    0.1a1