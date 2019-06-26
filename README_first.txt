1) First go at http://ti.arc.nasa.gov/tech/dash/pcoe/prognostic-data-repository/
Download files in the section called: "Turbofan Engine Degradation Simulation Data Set"
Those files datasets and also contain RUL_FD000x.txt which contains the Remaining Useful Life values (RUL) for x=1,2,3,4

2) Run evaluate_similarity_polygon_rulclipper.m
It contains the code within an example to reproduce RUL prediction for one instance in one testing dataset.
Modify it as necessary to make loop over all testing datasets: you should obtain around S=272 for the cumulated 
score for dataset #1. See the file "ex2_loop_rulclipper__not_runnable_readme.m".

3) If your are interested in computing the health indicators, then go in health_indicators_estimation_cmapss.m
[responses globalHI localHI] = health_indicators_estimation_cmapss(...)
Then you can use those HI for predictions. You can evaluate the impact of the features on the results.

For any questions, contact me at:
emmanuel.ramasso@femto-st.fr

If you use this code in any publication/project please cite [1]. 

[1] E. Ramasso, Investigating computational geometry for failure prognostics,
International Journal on Prognostics and Health Management, 5(5):1-18, 2014.

