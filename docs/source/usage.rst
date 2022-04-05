
.. _usage:

Usage 
============================


The following provide instructive examples on how to run MRF to generate forecasts and generalized time-varying parameters (GTVPs).

It is recommended that you see :ref:`Docs <docs>`, particularly the MRF module, before you proceed. You can proceed with the below as an example of how to get started. 

Python 
----------------------------

Implementation Example: Simple One-Step Forecasting
+++++++++++++++++++++++++++

First order of business is to import MRF and matplotlib, a useful plotting package:

.. code-block:: python

   from MRF import *
   import matplotlib.pyplot as plt

As a way to get started, we have included a dataset of simulated variables with the package at download time:

.. code-block:: python

   simulated_data = pd.read_csv("../Datasets/mrf_sim.csv", index_col = 'index')

We can take a look at this data using :code:`display(simulated_data.head(5))`::


   index    sim_y     sim_x1      sim_x2      sim_x3  ...    sim_x14    sim_x15   trend 
     0    -0.441805  1.262954   -1.045718   -0.390010 ...   0.095309   -0.276508    1 
     1     2.793370  -0.326233  -0.896211   -1.819222 ...   0.991170   -0.854418    2 
     2     2.537384  1.329799    1.269387    0.659181 ...   0.428252    1.484950    3 
     3     1.769591  1.272429    0.593841    0.459622 ...   1.118214   -1.597299    4 
     4     2.299628  0.414641    0.775634    1.616626 ...   -0.739658   0.374999    5 

Let's say we want to predict the last 50 observations. We can set up our oos_pos as follows:

.. code-block:: python

   oos_pos = np.arange(len(simulated_data) - 50 , len(simulated_data)) # lower should be oos start, upper the length of your dataset

Notice that our desired :math:`y_t` is in column position 0, so we will pass :code:`y_pos = 0`. Our desired :math:`X_t` are in index positions 1, 2 and 3, since we want our first 3 predictors to be time-varying, so we will pass :code:`x_pos = np.arange(1, 4)`. S_pos we will omit from our arguments, since we want all of our extra exogenous variables to be included in our overall predictor set.

If we want to speed things up, we can also select :code:`parallelise = True` and :code:`n_cores = 3` to run the code across 3 cores on our machine. 

.. warning::
   Running in parallel across all cores can cause your computer to temporarily slow down

The remaining hyperparameters we have chosen are relatively standard and the user should see :ref:`Docs <docs>` if they want to know more details.

Now we are ready to implement:

.. code-block:: python

   MRF = MacroRandomForest(data = simulated_data,
                           y_pos = 0,
                           x_pos = np.arange(1,4), 
                           B = 100, 
                           parallelise = True,
                           n_cores = 3,
                           resampling_opt = 2,
                           oos_pos = oos_pos,
                           trend_push = 4,
                           quantile_rate = 0.3, 
                           print_b = True,
                           fast_rw = True)

To get this running, we simply need to run the following command:

.. code-block:: python

   MRF_output = MRF._ensemble_loop()

Once our function has run through, we can access the output as a dictionary. For example, the following two commands will respectively return the forecasts and betas for the model.

.. code-block:: python

   forecasts = MRF_output['pred']
   betas = MRF_output['betas']

And we're done. You now have MRF predictions and GTVPs! Here's a look at our output:

Firstly, the predictions:

.. code-block:: python

   fig, ax = plt.subplots()
   plt.rcParams['figure.figsize'] = (20, 8)

   # Plotting actual versus original
   ax.plot(original_data['sim_y'].loc[oos_pos].shift(1), label = 'Actual', linewidth = 3, color ='mediumseagreen', linestyle = '--')
   ax.plot(forecasts, color = 'lightcoral', linewidth = 3, label = "MRF Ensemble")
   
   ax.legend(fontsize = 15)
   ax.set_ylabel("Value", fontsize = 15)
   ax.grid()
   ax.set_xlabel(r"$t$", fontsize = 16)
   ax.set_title("OOS predictions of MRF", fontsize = 15)

.. image:: /images/OOS_preds.png

And, last but not least, the GTVPs:

.. code-block:: python

   MRF.band_plots()

.. image:: /images/GTVPs.png


Implementation Example: Financial Trading
+++++++++++++++++++++++++++

To start with, let's read in one of our finance datasets:

.. code-block:: python

   data_in = pd.read_csv("../Datasets/finance.csv")

We can take a look at this data using :code:`display(data_in.head(5))`::


      Date     spy_close  spy_1d_returns   VIX_slope    yc_3m   yc_10y   yc_slopes_3m_10y   5Ewm     15Ewm      MACD    trend
   24/01/2013   1494.82      -0.002          -0.001     0.00     0.02        0.001         2.654     2.340    -11.071     1 
   25/01/2013   1502.96       0.005          -0.001     0.00     0.10        0.001         4.483     3.065    -12.489     2 
   28/01/2013   1500.18      -0.007          -0.002    -0.01     0.02        0.001         2.062     2.334    -12.216     3 
   29/01/2013   1507.84       0.007           0.002     0.00     0.03        0.001         3.928     3.000    -13.144     4 
   30/01/2013   1501.96      -0.009          -0.003     0.00     0.00        0.001         0.659     1.890    -11.913     5 
   
Since we are not going to predict the price, but rather the return, we need to assign our prices to a new variable (we will use it later) and remove it from our dataframe containing :math:`[y_t, X_t, S_t]`:

.. code-block:: python

   close_prices = data_in['spy_close']
   data_in = data_in.iloc[:, 1:]
   

We want to have a backtest (oos) period in order to evaluate MRF, so we are going to set up our out-of-sample period to include the last 350 observations:

.. code-block:: python

   oos_pos = np.arange(len(data_in[:-350]), len(data_in[:-1])+1)

Now for the MRF specification:

.. code-block:: python

   MRF = MacroRandomForest(data = data_in,
                           y_pos = 0,
                           x_pos = np.arange(1, 5), 
                           fast_rw = True, 
                           B = 50, 
                           mtry_frac = 0.3, 
                           resampling_opt = 2,
                           oos_pos = oos_pos, 
                           trend_push = 2,
                           quantile_rate = 0.3, 
                           parallelise = True)

And the MRF fitting:

.. code-block:: python

   mrf_output = MRF._ensemble_loop()

Now we can automatically evaluate the financial performance of MRF using the :code:`financial_evaluation()` function. This function will return 5 outputs: 1) The daily profit series associated with the induced strategy, 2) The cumulative profit series, 3) The annualised return, 4) The Sharpe ratio and 5) The maximum drawdown. These metrics are outlined in :ref:`Evaluation <fineval>`.

.. code-block:: python

   trading_statistics = MRF.financial_evaluation(model_forecasts = mrf_output['pred'], 
                                                 close_prices = close_prices)

   daily_profit = trading_statistics[0]
   cumulative_profit = trading_statistics[1]
   annualised_return = trading_statistics[2]
   sharpe_ratio = trading_statistics[3]
   maximum_drawdown = trading_statistics[4]

We can also get out a useful plot that compares the financial trading performance of MRF (green) versus 100 "monkey traders" that implement the same strategy (grey) and a "buy and hold" strategy on the S&P 500 (blue).

.. code-block:: python

   MRF.monkey_trader_plot(close_prices)

.. image:: /images/Trading.png

And voila, you have a fully trained and backtested model. You are ready to deploy your MRF-guided trading strategy.

R 
----------------------------


Implementation Example: Simple One-Step Forecasting
+++++++++++++++++++++++++++

As a way to get started, we can run a simulation to create a simple synthetic data set:

.. code-block:: r

   set.seed(0)
   data=matrix(rnorm(15*200),200,15)
   #DGP
   library(pracma)
   X=data[,1:3]
   y=crossprod(t(X),rep(1,3))*(1-0.5*I(c(1:200)>75))+rnorm(200)/2
   trend=1:200
   data.in=cbind(y,data,trend)

We can take a look at this data before proceeding. 

.. code-block:: r

   head(data.in)
       
   [1,] -0.4418048  1.2629543 -1.0457177 ...   0.09530868 -0.2765078   1
   [2,] -2.7933695 -0.3262334 -0.8962113  ...  0.99117035 -0.8544175   2
   [3,]  2.5373841  1.3297993  1.2693872  ...  0.42825204  1.4849503   3
   [4,]  1.7695908  1.2724293  0.5938409  ...  1.11821352 -1.5972987   4
   [5,]  2.2996275  0.4146414  0.7756343  ... -0.73965815  0.3749989   5
   [6,] -1.5550883 -1.5399500  1.5573704  ... -2.06393339  1.3272442   6

Let’s say we want to predict the last 50 observations. We can set up our oos_pos as follows:

.. code-block:: r

   oos_position = nrow(data.in)-50: nrow(data.in)

Once we have made our data set, we are ready to run MRF. We need to specify the position of our desired :math:`y_t`. In our case, this variable is in the first column, so we will set :code:`y.pos = 1`. Our desired :math:`X_t` are in index positions 1, 2 and 3, since we want our first 3 predictors to be time-varying, so we will pass :code:`x.pos = 2:4`. S_pos we will pass as :code:`s.pos = 2:ncol(data.in)`, since we want all of our extra exogenous variables to be included in our overall predictor set :math:`S_t`. 

The remaining hyperparameters we have chosen are relatively standard and the user should see :ref:`Docs <docs>` if they want to know more details.

.. code-block:: r

   mrf.output = MRF(data = data.in,
                    y.pos = 1,
                    x.pos = 2:4,
                    S.pos = 2:ncol(data.in),
                    oos.pos = oos_position,
                    mtry.frac = 0.25, 
                    trend.push = 4,
                    quantile.rate = 0.3, 
                    B = 100)

And we're done. You now have MRF predictions and GTVPs! Here's a look at our output:

.. image:: /images/R_GTVPs.svg
      

Implementation Example: One-Step Macro Forecasting
+++++++++++++++++++++++++++

Let's say that our goal is to forecast non-farm payrolls one period ahead using the FRED macroeconomic data base (FREDMD).

Let's firstly load MRF. We will also load the fbi package, which let's us read and manipulate FRED data, and several other useful libraries. 

.. code-block:: r

   library(MacroRF)
   library(fbi)
   library(tidyverse)
   library(lubridate)
   library(vars)
   library(pracma)

We are also going to initialise the select method, which comes from the dplyr package. This will be useful for data manipulation:

.. code-block:: r
   
   select <- dplyr::select

With the boring stuff out of the way, let's go about creating our forecasting setup. 
   
Our goal is to forecast non-farm payrolls, so we'll set that as our dependent variable. As predictors, we're going to have 5 factors of the FREDMD data base with the first three (our :math:`X_t`) included in our linear equation, all at a lag of one. Our data is going to start on Jan 1st 2003 and we're going to make predictions on a one-period forecast horizon:

.. code-block:: r

   ### Dependent variable from FRED
   my_var <- "PAYEMS"   

   ### Number of factors
   my_k <- 5

   ### First number of factors in linear eqn
   my_x <- 3

   ### Lags
   my_p <- 1

   ### Start Date
   start_date <- "2003-01-01"

   ### Forecast Horizon
   hor <- 1

With our forecasting setup defined, let's read the data from FRED:

.. code-block:: r

   # Reading the data from FRED
   df <- fredmd(file = "https://files.stlouisfed.org/files/htdocs/fred-md/monthly/current.csv",
               transform = TRUE,
               date_start = ymd(start_date))
   
   # Reading column names from FRED
   df_for_names <- read_csv("https://files.stlouisfed.org/files/htdocs/fred-md/monthly/current.csv")

Taking a look at the data frame, we have 229 rows and 127 columns (not all shown here):

.. code-block:: r

   print(head(df))

             RPI        W875RX1     DPCERA3M086SBEA  ...        INVEST    VIXCLSx
   529 -0.0032978454 -0.004065960   -0.0001315782    ...    -0.020117881  30.6685
   530 -0.0037021507 -0.003959223   -0.0032350855    ...    -0.002235762  35.1947
   531  0.0017066104  0.001560944    0.0057321149    ...    -0.002235762  35.1947
   532  0.0046942035  0.004801033    0.0047141822    ...     0.001445046  27.1423
   533  0.0077470739  0.007832646    0.0032133589    ...     0.009581121  22.5485
   534  0.0035093161  0.003418945    0.0053366834    ...    -0.002602376  22.3490
   535  0.0009887095  0.000777240    0.0045115509    ...    -0.017077098  21.2068
Let's process the data, including handling outliers and missing values:

.. code-block:: r

   # Setting column names
   colnames(df) <- colnames(df_for_names)

   # Removing outliers in the series
   df <- rm_outliers.fredmd(df)

   df[["sasdate"]] <- NULL

   # Handling missing values
   imputed <- tw_apc(X = df,
             kmax = my_k,
             center = TRUE,
             standardize = TRUE)
   

Let's set up our matrix of factors using principal component analysis (PCA):

.. code-block:: r

   # Decomposing the data matrix into sparse, low-rank components
   afm <- rpca(X = imputed[["data"]], 
            kmax = my_k,
            standardize = TRUE)

   # Establishing and scaling robust PCA factors - the variables for our forecast
   Fmat <- prcomp(scale(imputed[["data"]]), rank. = my_k)$x

   # Encoding the predictors
   ma_mat <- embed(scale(imputed[["data"]]), 60)

   # Merge the matrices
   ma_mat <- cbind(scale(imputed[["data"]]) %>% tail(nrow(ma_mat)), ma_mat)

   # Decomposing the data matrix into sparse, low-rank components
   MAFmat <- prcomp(ma_mat, rank. = my_x)$x

Let's set up our variables for easy access:

.. code-block:: r

   set.seed(1234)  
   n <- nrow(MAFmat)
   idx <- which(colnames(df) == my_var)
   X <- imputed[["data"]][, idx]
   X <- tail(X, n)
   Fmat <- tail(Fmat, n)
   Y <- cbind(X, Fmat, MAFmat)
   colnames(Y) <- c(my_var, paste0("F_", 1:my_k), paste0("MAF_", 1:my_x))

We can now take a look at our input data:

.. code-block:: r

   print(Y)

           PAYEMS       F_1         F_2         F_3           F_4          F_5        MAF_1     MAF_2     MAF_3
   60  0.0007806966  -3.448621  -3.7578079   2.135086615   6.1580987  -0.75658675  -24.43069  23.65243  -11.18031
   61  0.0000794812  -2.437831   1.5382544  -1.779136678   9.9564912  -0.70590524  -25.74333  23.10433  -11.57520
   62 -0.0005709598  -5.140423   0.2617188  -1.144619273   7.8978095  -0.52537640  -27.53283  22.53457  -12.68836
   63 -0.0003543035  -4.333899   3.1338272  -1.938025976   8.5230994  -0.20404637  -29.39276  21.75854  -13.35939
   64 -0.0017371797  -4.135100   0.6067619  -0.008076702  -0.9087045  -1.57366593  -31.23286  21.07104  -14.41252
   65 -0.0012831063  -1.806275   3.6440667  -2.393721847  -3.3302690  -0.02333614  -32.65311  20.01826  -14.79434


And with all of that out of the way, it's time to fit MRF! We're going to loop through from 1 until the eventual forecast horizon, each time setting our data matrix and the position of our variables that we want to be time-varying.

.. code-block:: r

      Y_temp <- Y[c(1:nrow(Y), nrow(Y)), ]

      mat <- VAR(Y_temp, p = i + my_p - 1, type = "trend")[["datamat"]] %>%
         as.data.frame() %>%
         select(my_var, contains(".l"), trend)

      rownames(mat) <- NULL

      x_pos1 <- which(str_detect(colnames(mat), paste0("F_", 1:my_x, ".l", rep(1:my_p, each = my_x), collapse = "|")))
      x_pos2 <- which(str_detect(colnames(mat), paste0(my_var, ".l", i, collapse = "|")))
      x_pos = c(x_pos1, x_pos2)

      model <- MRF(mat, x.pos = x_pos,
                        oos.pos = nrow(mat),
                        ridge.lambda = 0.30,
                        trend.push = 6,
                        B = 250,
                        quantile.rate = 0.3,
                        fast.rw = TRUE)

That's it! Our models are fit and the training is finished. All we need to do now is to access our predictions.

.. code-block:: r

   preds <- model[["pred"]]

   y <- 149629 * cumprod(exp(preds)) - 149629 # Our final forecast!

   print(y)
   
   [1] 530.0887

And there we have it, our final forecasted value is 530.0887. If we want, we can also access the pre-ensembled forecasts:

.. code-block:: r

   d <- 149629 * exp(model$pred.ensemble) - 149629
   d_df <- data.frame(d)

Let's visualise the range of our pre-ensembled forecasts by plotting the distribution:

.. code-block:: r

   ggplot(d_df) +
   theme_bw() +
   aes(x = d) +
   geom_density(adjust = 2,fill = "grey") +
   xlim(c(0, 1000)) +
   geom_vline(xintercept = median(d)) +
   theme(plot.background = element_rect(fill = "transparent", colour = NA))+
   ggtitle("Distribution (density) of pre-ensembled forecasts") +
   theme(plot.title = element_text(hjust = 0.5)) +
   xlab("Forecast") 

.. image:: /images/distplot.png

We can also look at the GTVPs to visualise the change in the coefficients corresponding to the constant (:math:`\beta_0`, top-left), the first principal component (:math:`\beta_1`, top-right), second principal component (:math:`\beta_2`, bottom-left) and the third principal component (:math:`\beta_3`, bottom-right).

.. image:: /images/GTVP_nfp.svg


Implementation Example: Multi-Step Macro Forecasting
+++++++++++++++++++++++++++

Let's say that our goal is to forecast inflation (CPI) three periods ahead using the FRED macroeconomic data base (FREDMD).

Firstly, we will need to load MRF. We will also load the fbi package, which let's us read and manipulate FRED data, and several other useful libraries.

.. code-block:: r

   library(MacroRF)
   library(fbi)
   library(tidyverse)
   library(lubridate)
   library(vars)


Our goal is to forecast CPI, so we'll set that as our dependent variable. As predictors, we're going to have 5 factors (principal components) of the FREDMD data base with the first three (our :math:`X_t`) included in our linear equation, all at a lag of three periods. Our data is going to start on Jan 1st 2003 and we're going to make predictions on a three-period forecast horizon:

.. code-block:: r

   ### Variable from FRED
   my_var <- "CPIAUCSL"

   ### Number of factors
   my_k <- 5

   ### First number of factors in linear eqn
   my_x <- 3

   ### Lags
   my_p <- 3

   ### Start Date
   start_date <- "2003-01-01"

   ### Forecast Horizon
   hor <- 3

With our forecasting setup defined, let’s read the data from FRED:

.. code-block:: r

   df <- fredmd(file = "https://files.stlouisfed.org/files/htdocs/fred-md/monthly/current.csv",
               transform = TRUE,
               date_start = ymd(start_date))

   df_for_names <- read_csv("https://files.stlouisfed.org/files/htdocs/fred-md/monthly/current.csv")


Taking a look at the data frame, we have 229 rows and 127 columns (not all shown here):

.. code-block:: r

   print(head(df))

             RPI        W875RX1     DPCERA3M086SBEA  ...        INVEST    VIXCLSx
   529 -0.0032978454 -0.004065960   -0.0001315782    ...    -0.020117881  30.6685
   530 -0.0037021507 -0.003959223   -0.0032350855    ...    -0.002235762  35.1947
   531  0.0017066104  0.001560944    0.0057321149    ...    -0.002235762  35.1947
   532  0.0046942035  0.004801033    0.0047141822    ...     0.001445046  27.1423
   533  0.0077470739  0.007832646    0.0032133589    ...     0.009581121  22.5485
   534  0.0035093161  0.003418945    0.0053366834    ...    -0.002602376  22.3490
   535  0.0009887095  0.000777240    0.0045115509    ...    -0.017077098  21.2068

Let's process the data, including handling outliers and missing values:


.. code-block:: r

   # Setting column names
   colnames(df) <- colnames(df_for_names)

   # Removing outliers in the series
   df <- rm_outliers.fredmd(df)

   df[["sasdate"]] <- NULL

   # Handling missing values
   imputed <- tw_apc(X = df,
            kmax = my_k,
            center = TRUE,
            standardize = TRUE)

   
Let’s set up our matrix of factors using principal component analysis (PCA):

.. code-block:: r

   # Decomposing the data matrix into sparse, low-rank components
   afm <- rpca(X = imputed[["data"]], 
               kmax = my_k,
               standardize = TRUE)

   # Establishing and scaling robust PCA factors - the variables for our forecast
   Fmat <- prcomp(scale(imputed[["data"]]), rank. = my_k)$x

   # Encoding the predictors
   ma_mat <- embed(scale(imputed[["data"]]), 12)

   # Merge the matrices
   ma_mat <- cbind(scale(imputed[["data"]]) %>% tail(nrow(ma_mat)), ma_mat)

   # Decomposing the data matrix into sparse, low-rank components
   MAFmat <- prcomp(ma_mat, rank. = my_x)$x

Let’s set up our variables for easy access:

.. code-block:: r

   n <- nrow(MAFmat)
   idx <- which(colnames(df) == my_var)
   X <- imputed[["data"]][, idx]
   X <- tail(X, n)
   Fmat <- tail(Fmat, n)
   Y <- cbind(X, Fmat, MAFmat)

We can now take a look at our input data:

.. code-block:: r

   print(Y)

         CPIAUCSL      F_1         F_2         F_3          F_4         F_5       MAF_1      MAF_2      MAF_3
   12  2.158370e-03  1.173100   0.1729750  -3.42070409  -1.3607214  -2.0993337  -4.068628  4.7082964  -13.40729
   13  1.604339e-03  2.049119   0.7857849  -3.07097371  -0.7735698  -1.8544508  -4.226717  3.9713696  -13.63371
   14 -2.158623e-03  1.074777  -2.8700708  -0.03065825  -0.7579336  -2.6595706  -5.273620  3.1579670  -12.32650
   15 -4.590208e-06  1.588660  -2.6480670  -1.28308755   0.1182666  -1.7615131  -6.650085  2.6910707  -10.96458
   16 -5.380462e-04  1.728049  -3.8522863  -1.42536771  -4.0382254  -0.6121978  -8.065427  1.6399727  -10.62753
   17  2.657721e-03  3.827819  -1.4023308  -1.44571729  -5.2314203  -1.8990988  -8.841191  0.2838795  -11.98170

We're going to want to save our forest output as we loop through to the eventual forecast horizon, so we'll create an array where the output can be stored. We can also set the seed for replicability:

.. code-block:: r

   r_list <- list()
   set.seed(1234)

And with all of that out of the way, it's time to fit MRF! We're going to conduct recursive forecasting by looping through from 1 until the eventual forecast horizon, each time setting our data matrix and the position of our variables that we want to be time-varying:

.. code-block:: r

   for(i in 1:hor) {

   if(i == 1) {
      Y_temp <- Y[c(1:nrow(Y), nrow(Y)), ]
      mat <- VAR(Y_temp, p = i + my_p - 1, type = "trend")[["datamat"]] %>%
         as.data.frame() %>%
         select(my_var, contains(".l"), trend)

      rownames(mat) <- NULL
      x_pos1 <- which(str_detect(colnames(mat), paste0("F_", 1:my_x, ".l", rep(1:my_p, each = my_x), collapse = "|")))
      x_pos2 <- which(str_detect(colnames(mat), paste0(my_var, ".l", i, collapse = "|")))
      x_pos = c(x_pos1, x_pos2)
      r_list[[i]] <- MRF(mat, x.pos = x_pos,
                        oos.pos = nrow(mat),
                        ridge.lambda = 0.30,
                        rw.regul = 0.80, 
                        trend.push = 6,
                        B = 40,
                        fast.rw = TRUE)
   } 
   
   else if(i > 1) {
      Y_temp <- Y[c(1:nrow(Y), rep(nrow(Y), i)), ]
      sel_rm <- paste0(".l", 1:(i-1), collapse = "|")
      mat <- VAR(Y, p = i + my_p, type = "trend")[["datamat"]] %>%
         as.data.frame() %>%
         select(my_var, contains(".l"), trend) %>% 
         select(-matches(sel_rm))
      rownames(mat) <- NULL
      mat[,1]=mat[,1]*1200
      
      x_pos1 <- which(str_detect(colnames(mat), paste0("F_", 1:my_x, ".l", rep(i:(my_p + i), each = my_x), collapse = "|")))
      x_pos2 <- which(str_detect(colnames(mat), paste0(my_var, ".l", i, collapse = "|")))
      r_list[[i]] <- MRF(mat, x.pos = x_pos,
                        oos.pos = nrow(mat),
                        ridge.lambda = 0.30,
                        rw.regul = 0.80, 
                        trend.push = 6 + i,
                        B = 40)
      
   }
   }

That's it! Our models are fit and the training is finished. All we need to do now is to assign our predictions for easy access.

.. code-block:: r

   preds <- c()
   for(i in 1:hor) preds[i] <- r_list[[i]][["pred"]]
   preds <- sapply(r_list, function(x) x[["pred"]])

   print(preds)
   [1] -0.00156105  0.52877814  0.79409874

Now we have our raw three-step predictions. All that's left to do is to convert that back to the original CPI units.