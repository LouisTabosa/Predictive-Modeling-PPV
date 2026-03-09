function [fitresult, gof] = createFit(Distance_m_, Charge_Kg_, PVS_mm_s_)
%CREATEFIT(DISTANCE_M_,CHARGE_KG_,PVS_MM_S_)
%  Create a fit.
%
%  Data for 'USBM' fit:
%      X Input: Distance_m_ from Data_200
%      Y Input: Charge_Kg_ from Data_200
%      Z Output: PVS_mm_s_ from Data_200
%  Output:
%      fitresult : a fit object representing the fit.
%      gof : structure with goodness-of fit info.
%
%  See also FIT, CFIT, SFIT.



%% Fit: 'USBM'.
[xData, yData, zData] = prepareSurfaceData( Distance_m_, Charge_Kg_, PVS_mm_s_ );

% Set up fittype and options.
ft = fittype( 'a*(x/(y^(1/2)))^b', 'independent', {'x', 'y'}, 'dependent', 'z' );
opts = fitoptions( 'Method', 'NonlinearLeastSquares' );
opts.Display = 'Off';
opts.StartPoint = [0.743132468124916 0.392227019534168];

% Fit model to data.
[fitresult, gof] = fit( [xData, yData], zData, ft, opts );

% Plot fit with data.
figure( 'Name', 'USBM' );
h = plot( fitresult, [xData, yData], zData );
legend( h, 'USBM', 'PVS_mm_s_ vs. Distance_m_, Charge_Kg_', 'Location', 'NorthEast', 'Interpreter', 'none' );
% Label axes
xlabel( 'Distance_m_', 'Interpreter', 'none' );
ylabel( 'Charge_Kg_', 'Interpreter', 'none' );
zlabel( 'PVS_mm_s_', 'Interpreter', 'none' );
grid on
% Display R-squared and RMSE on the figure
rsq = gof.rsquare;
rmse = gof.rmse;
txt = sprintf('R^{2} = %.4f\nRMSE = %.4f', rsq, rmse);
% Place textbox in upper-left corner of the axes
ax = gca;
% Use normalized units so placement is consistent
annotation('textbox',[0.15 0.75 0.2 0.1], 'String', txt, 'FitBoxToText','on', ...
    'BackgroundColor','white','EdgeColor','black','FontSize',10);