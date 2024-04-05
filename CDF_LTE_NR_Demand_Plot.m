% Disclaimer: NIST-developed software is provided by NIST as a public service. You may use, copy, and distribute copies of the software in any medium, 
% provided that you keep intact this entire notice. You may improve, modify, and create derivative works of the software or any portion of 
% the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed 
% the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards 
% and Technology as the source of the software. 
% 
% NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT, OR ARISING BY 
% OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT, 
% AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY 
% DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING 
% BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.
% 
% You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated 
% with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, 
% programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a 
% failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection 
% within the United States.

% Objective: Code to plot the cdf of LTE and NR demand data 
clc;
clear all;
close all;

% load the statistics obtained from the multiple instances of demand process generated using ARMA process (see code: Generating_Demand_ARMA.m) 
demand_lte_hr = readtable("Output_2115_hourly.tsv", "FileType","text",'Delimiter', '\t');
demand_nr_hr = readtable("Output_2115_hourly_NR.tsv", "FileType","text",'Delimiter', '\t');
demand_lte_min = readtable("Output_2115_min.tsv", "FileType","text",'Delimiter', '\t');
demand_nr_min = readtable("Output_2115_min_NR.tsv", "FileType","text",'Delimiter', '\t');

d_lte_hr = demand_lte_hr.NRB;
d_nr_hr = demand_nr_hr.NRB;
d_lte_min = demand_lte_min.NRB;
d_nr_min = demand_nr_min.NRB;

h = cdfplot(d_lte_hr);
set(h,'LineWidth',6);
hold on;
h = cdfplot(d_nr_hr);
set(h,'LineWidth',6);
hold off;
set(gca,'FontSize',48,'FontWeight','bold');
title('')
xlabel('Average PRB Demand','Fontweight','bold','Fontsize',48,'Interpreter','latex')
ylabel('CDF','Fontweight','bold','Fontsize',48,'Interpreter','latex')
set(gca, 'TickLabelInterpreter', 'latex');
legend('LTE-Compiled','NR-Synthetic','Location','best','Fontsize',38,...
    'Fontweight','bold');
set(gcf,'Color','w');
set(gcf,'Position',get(0,'ScreenSize'));
fig_name = sprintf('CDF_LTE_NR_Demand_hour.pdf');
addpath 'Export_fig'
export_fig(fig_name);
close;

h = cdfplot(d_lte_min);
set(h,'LineWidth',6);
hold on;
h = cdfplot(d_nr_min);
set(h,'LineWidth',6);
hold off;
set(gca,'FontSize',48,'FontWeight','bold');
title('')
xlabel('Average PRB Demand','Fontweight','bold','Fontsize',48,'Interpreter','latex')
ylabel('CDF','Fontweight','bold','Fontsize',48,'Interpreter','latex')
set(gca, 'TickLabelInterpreter', 'latex');
legend('LTE-Compiled','NR-Synthetic','Location','best','Fontsize',38,...
    'Fontweight','bold');
set(gcf,'Color','w');
set(gcf,'Position',get(0,'ScreenSize'));
fig_name = sprintf('CDF_LTE_NR_Demand_min.pdf');
addpath 'Export_fig'
export_fig(fig_name);
close;
