.. _main:

Macroeconomic Random Forest
===========================

.. image:: /images/MaRF_logo.svg

.. raw:: html

   <embed>
        <p align="center">
            <a href="https://www.r-project.org/">
                <img src="./_static/svg/R_button.svg"
                    width="77" height="12" alt="R"></a> &nbsp;
            <a href="https://www.python.org/">
                <img src="./_static/svg/Python_button.svg"
                    width="100" height="30" alt="python"></a> &nbsp;
            <a href="">
                <img src="./_static/svg/MRF_button.svg"
                    alt="MRF" width="108" height="30"></a> &nbsp;
            <a href="">
                <img src="./_static/svg/Maintained_button.svg"
                    alt="Maintained?" width="135" height="30"></a> &nbsp;
        </p>
   </embed>


.. raw:: html

   <embed>
        <p align="center">
            

        </p>
   </embed>

Ever wanted the power of a Random Forest with the interpretability of a Linear Regression model? Well now you can...

Created by Ryan Lucas, this code base is the official open-source implementation of "The Macroeconomy as a Random Forest (MRF)" by Philippe Goulet Coulombe. MRF is a time series modification of the canonical Random Forest Machine Learning algorithm. It uses a Random Forest to flexibly model time-varying parameters in a linear macro equation. This means that, unlike most Machine Learning methods, MRF is directly interpretable via its main output - what are known as Generalised Time Varying Parameters (GTVPs). 
  
The model has also shown forecasting gains over numerous alternatives and across many time series datasets. It is well suited to macroeconomic forecasting, but there are also many possible extensions to quantitative finance, or any field of science with time series data. The full paper corresponding to the implementation can be found here: https://arxiv.org/abs/2006.12724


Installation
============

Python
------

To install the package in Python, you can simply do:

.. code-block:: python

   git clone https://github.com/RyanLucas3/MacroRandomForest


Now you're ready to get started! Head over to :ref:`Usage <usage>` to find out how to implement MRF.


R
---

To install the package in R, simply run the following commands:

.. code-block:: r

   install.packages('devtools') 
   library(devtools)
   install_github("philgoucou/macrorf");
   library(MacroRF)

.. note::
   This documentation contains detailed information about the implementation of the Python version, with less detail for R. 
   
   Still, the implementation is simple and the same guidelines for hyperparameter selection will apply to both versions. Head over to  :ref:`Usage<usage>` for more details.



.. toctree::
   :maxdepth: 2
   :caption: Contents:

   how_it_works
   modules
   usage


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
