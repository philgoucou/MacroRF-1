Evaluation module
=================

This module is used to evaluate the predictions of MRF. The following functions are called by the statistical_evaluation() and financial_evaluation() methods of the MRF module.

Statistical Evaluation Metrics
--------------------------------
As statistical evaluation metrics, we use the standard MAE and MSE:

.. math::

   \begin{align}
   MAE = \frac{1}{\vert T\vert} \sum_{t \in T} |\hat{y}_{t+k|t} - y_{t+k}| \\
   MSE = \frac{1}{\vert T\vert} \sum_{t \in T} (\hat{y}_{t+k|t} - y_{t+k})^2
   \end{align}

Where :math:`k` is the forecast horizon and :math:`\vert T\vert` is the cardinality of the index set :math:`T`. In our case this index set is the specified out-of-sample observations. 

.. _fineval:
Financial Evaluation Metrics
--------------------------------

.. note::
   You should only use financial_evaluation() if your target variable is a financial return and you can provide the underlying tradable asset.

   This means the financial evaluation functions will be next to meaningless for macro forecasting.

As a method for the financial evaluation of MRF predictions, we use a trading strategy described in [1]_ to generate binary long/short market signals. This strategy is agnostic to the forecast horizon used. Our trading signal :math:`S_t` becomes a weighted average of directional (up/down) signals obtained by our model.

.. math::

   \begin{align}
   S_{t}:=\frac{1}{k} \times \sum_{j=0}^{k-1}\left(\mathbb{1}\left[\hat{y}_{t+k-j \mid t-j}>0\right]-\mathbb{1}\left[\hat{y}_{t+k-j \mid t-j}<0\right]\right)
   \end{align}


Given :math:`r_t` as the daily profit associated with the trading strategy and :math:`T_{prof}`  as the index for profit calculation, with :math:`T_{end}` as the last index in :math:`T_{prof}`, we can calculate Annualised Return as follows:

.. math::

   \begin{align}\label{pi_t}
   \Upsilon_{t} = \sum_{\tau \in [t]} r_\tau
   \end{align}

.. math::

   \begin{align}\label{Annualised}
   ANR = \frac{252 \times \Upsilon_{T_{end}}}{\vert T_{prof} \vert} 
   \end{align} 

Mean return :math:`\bar{r}` and Sharpe Ratio :math:`SR` are then calculated as follows:

.. math::
    \begin{align}
        \bar{r} = \frac{1}{{|T_{prof}|}} \times \sum_{t \in T_{prof}} r_t \\
        SR = \sqrt{252} \times \frac{\bar{r}}{St Dev(\{r_t\}_{t \in T_{prof}})}
    \end{align}

Maximum drawdown, :math:`MDD`, measures the maximum observed loss from a peak to a trough in the value of a holding. Representing the value of the portfolio to be :math:`\Pi_t` = :math:`\Upsilon_t + 1`, :math:`MDD` is calculated as follows:

.. math::

   \begin{align} \label{MDD}
      MDD = \underset{t\in T_{prof}} \min \left\{  \frac{\Pi_t}{ \underset{\tau \in [t]} \max \left\{ \Pi_{\tau}\right\}} - 1\right\}
   \end{align}


Code
----

.. automodule:: Evaluation
   :members:
   :undoc-members:
   :show-inheritance:

References
----------

.. [1] Ruogu Yang, Parley. Lucas, Ryan. Schelpe, Camilla. (2021). Adaptive Learning on Time Series: Method and Financial Applications. arXiv preprint arXiv: 2110.11156.

