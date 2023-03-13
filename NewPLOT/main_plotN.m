function [] = main_plotN()
    close all;
    % Markers = {'o','+','*','.','x','_','|','square','diamond','^','v','>','<','pentagram' }
    % To change X and Y axis properties GOTO LINE 44
    % To change dataset : Go to line 65
    %Relativer error LINE 219
    RUNS = 3;
    EPOCHS = 15;
    lambdas = [1e-5];%1e0 1e-2 1e-4];

    etas = [ 1 1e-1 1e-2 1e-3];%
    rhos = [ 1];%
    d = 1; %dataet number from followimg list

   % deltas = [100 1e1 1e0 1e-1 1e-2 1e-3 1e-4 1e-5];
   deltas=[1 0.1 0.01 0.001 ];%0.1 0.01];
    COLS = [2000];%[-1 10 50 100 500];
    BSS = [128];
    
    %addpath('/home/hardik/Desktop/')
    path = 'Nystrom_Result23/RSN/';%'Nystrom_Result23/';
%     path = 'result_s2qn/';
    
    datasets = {         % COL
        'REALSIM'   %1
        'CIFAR10B'  %2
        'MNISTB'    %3
        'EPSILON'   %4
        'ADULT'     %5 ---> 10
        'W8A'       %6 ---> 20
        'ALLAML'    %7
        'GISETTE'   %8 ---> 50 
        'MRI'       %9
        'IJCNN'     %10
        };

    lw = 1.8;%RUNS;
    ms = 6;
    params = initN(lw, ms, lambdas, etas, rhos, RUNS, EPOCHS, COLS(1), BSS(1),deltas);
    
     sparams = {
% params('NEWTON')
% params('NG')
% params('RNG')
% params('SGD')
% params('RNGS')
params('LBFGS')
params('NGD')
params('NGD1')
params('NGD2')
params('RSN')
% params('Nystrom_GDLM')
% params('Nystrom_GDLM1')
% params('Nystrom_GDLM2')
%params('LBFGS')
%         params('NSVRG')
params('SVRG-LBFGS')
params('SVRG-SQN')
%            params('SQN')
%            params('SVRG')
params('adam')
% params('NSGD')
%         params('NSVRG')      
        };
    
    yparams = {'cost', 'val_cost', 'acc_tr', 'acc_val','gnorm'};
    xparams = {'time', 'epoch'};
    plot_params.sort = yparams{1};
    plot_params.y = yparams{5};
    plot_params.x = xparams{1};
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %To change the data enter the number from the data list of that dataset
    
    for dsi=d %6
            disp(datasets{dsi})
    for l = 1 :length(lambdas)
%          sparams{9}.etas=etas(l);
%          sparams{1}.etas=etas(4);
         %sparams{2}.rhos=rhos(l);
         %subplot(1,4,l);

        figure;
        plot_method_lambda(strcat(path, datasets{dsi}, '/'), sparams, lambdas(l), plot_params, 0*1e-5);
%         xlim([0,200]);
%         saveas(gcf,strcat('s4qn_',lower(datasets{dsi}),'_vtc_lam',num2str(l+1),'.eps'),'epsc');
 %       legendmarkeradjust(20,20);
        fprintf('\n');
    end
    end
%    legend({'S4QN-N', 'S4QN-F', 'SVRG-LBFGS', 'OBFGS', 'SVRG-SQN', 'SQN', 'ADAM', 'NSGD-D', 'NSGD', 'NSVRG', 'NSVRG-D'})
% legend({'S4QN-N', 'S4QN-F', 'NSVRG-D', 'NSGD'});
end

function plot_method_lambda(dataset, sparams, lambda, plot_params, ref)
    hold on;
    set(gca, 'FontSize', 18);
    title(strcat('\lambda=',sprintf('10^{%0.0f}', log10(lambda))));
    switch plot_params.x
        case 'time'
            xlabel('Time (seconds)');
        case 'epoch'
            xlabel('Epochs');
    end
    switch plot_params.y
        case 'cost'
            ylabel('Opt. Error (log scale)');
        case 'val_cost'
            ylabel('Test Error (log scale)');
        case 'acc_tr'
            ylabel('Train Accuracy');
        case 'acc_val'
            ylabel('Test Accuracy');
        case 'gnorm'
            ylabel('$\| \nabla f(w) \|$','Interpreter','latex');
    end
    if ref>=0
        opt_cost=inf;
        for m = 1:length(sparams)
            hold on;
            [bestmu, bestsg, besteta, bestrho, bestdelta] = find_bestN(dataset, sparams{m}.name, lambda, sparams{m}.etas, sparams{m}.rhos, sparams{m}.EPOCHS, sparams{m}.RUNS, sparams{m}.COL, sparams{m}.BS, plot_params.sort,sparams{m}.deltas);
            if bestmu.Count > 1
                y = bestmu(plot_params.y);
                if strcmp(plot_params.y, 'cost') || strcmp(plot_params.y, 'val_cost') || strcmp(plot_params.y,'gnorm')
                    opt_cost = min(opt_cost,min(y));
                end
            end
        end
    end
    maxy = -inf;
    miny = inf;
    for m = 1:length(sparams)
        hold on;
        [bestmu, bestsg, besteta, bestrho, bestdelta] = find_bestN(dataset, sparams{m}.name, lambda, sparams{m}.etas, sparams{m}.rhos, sparams{m}.EPOCHS, sparams{m}.RUNS, sparams{m}.COL, sparams{m}.BS, plot_params.sort,sparams{m}.deltas);
        if bestmu.Count > 1
            if strcmp(plot_params.y, 'val_cost') || strcmp(plot_params.y,'acc_val')
                x=bestmu(plot_params.x);
                x=x(2:end);
            else
                x=bestmu(plot_params.x);
            end
            if ref>=0 && (strcmp(plot_params.y, 'cost') || strcmp(plot_params.y, 'val_cost')) % || strcmp(plot_params.y,'gnorm'))
                y = (bestmu(plot_params.y)-opt_cost+ref)/(1+opt_cost);
                s = bestsg(plot_params.y)/(1+opt_cost);
            else
                y = bestmu(plot_params.y);
                s = bestsg(plot_params.y);
            end
            if bestrho == -1
                displayname=strcat(strrep(sparams{m}.name, '_', '-'), ' (\eta=', sprintf('10^{%0.0f})',log10(besteta)));
            elseif bestrho == 0
                displayname=strcat(strrep(sparams{m}.name, '_', '-'), ' (\eta=', sprintf('10^{%0.0f},',log10(besteta)),' \rho=\mid\midZ\mid\mid_{F})');
            else
                displayname=strcat(strrep(sparams{m}.name, '_', '-'), ' (\eta=', sprintf('10^{%0.0f},',log10(besteta)),' \rho=', sprintf('10^{%0.0f},', log10(bestrho)),' \delta=', sprintf('10^{%0.0f})', log10(bestdelta)));
            end
            if strcmp(plot_params.y, 'cost') || strcmp(plot_params.y, 'val_cost') || strcmp(plot_params.y,'gnorm')
                [~, idx] = min(y);
                if bestrho == -1
                    fprintf('%-23s: lambda: %.1e  eta: %.1e  %-12s  %s: %0.6f  @(idx: %d, val: %d)\n', sparams{m}.name, lambda, besteta, ' ', plot_params.y, y(idx), idx, round(x(idx)));
                else
                    fprintf('%-23s: lambda: %.1e  eta: %.1e  rho: %.1e delta: %.1e %s: %0.6f  @(idx: %d, val: %d)\n', sparams{m}.name, lambda, besteta, bestrho, bestdelta, plot_params.y, y(idx), idx, round(x(idx)));
                end
                maxy = max(maxy,y(1));
                miny = min(miny,min(y));
            else
                [~, idx] = max(y);
                if bestrho == -1
                    fprintf('%-23s: lambda: %.1e  eta: %.1e  %-12s  %s: %0.6f  @(idx: %d, val: %d)\n', sparams{m}.name, lambda, besteta, ' ', plot_params.y, y(idx), idx, round(x(idx)));
                else
                    fprintf('%-23s: lambda: %.1e  eta: %.1e  rho: %.1e delta: %.1e %s: %0.6f  @(idx: %d, val: %d)\n', sparams{m}.name, lambda, besteta, bestrho, bestdelta, plot_params.y, y(idx), idx, round(x(idx)));
                end
                maxy = max(maxy,max(y));
                miny = min(miny,min(y));
            end
            idx=length(y);
            if length(sparams)==1
                displayname = strcat(displayname, '@\lambda=', sprintf('10^{%0.0f})', log10(lambda)));
               % errorbar(x(1:idx), y(1:idx), s(1:idx), 'markersize', sparams{m}.markersize, 'linewidth', sparams{m}.linewidth, 'MarkerFaceColor', sparams{m}.facecolor, 'displayname', displayname);
                 plot(x(1:idx), y(1:idx), 'markersize', sparams{m}.markersize, 'linewidth', sparams{m}.linewidth, 'MarkerFaceColor', sparams{m}.facecolor, 'displayname', displayname);
            else
               % errorbar(x(1:idx), y(1:idx), s(1:idx), 'linestyle', sparams{m}.line, 'color', sparams{m}.linecolor, 'Marker', sparams{m}.marker, 'markersize', sparams{m}.markersize, 'linewidth', sparams{m}.linewidth, 'MarkerFaceColor', sparams{m}.facecolor, 'displayname', displayname);
                plot(x(1:1:idx), y(1:1:idx), 'linestyle', sparams{m}.line, 'color', sparams{m}.linecolor, 'Marker', sparams{m}.marker, 'markersize', sparams{m}.markersize, 'linewidth', sparams{m}.linewidth, 'MarkerFaceColor', sparams{m}.facecolor, 'displayname', displayname, 'MarkerIndices', 1:idx);
            end
        end
    end
    if maxy > miny 
        ylim([miny, maxy]);
    end
    %l = legend('FontSize',18,'Orientation','horizontal','NumColumns',3);
    %get(legend)
    %l.Orientation = 'horizontal';
    %legendmarkeradjust(20,20);

    if strcmp(plot_params.y, 'cost') || strcmp(plot_params.y, 'val_cost')  || strcmp(plot_params.y,'gnorm')
        set(gca, 'YScale', 'log');
    end
end
