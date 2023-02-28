function [w, infos] = rngd(problem, in_options,s)
% Netwon's method.
%
% Inputs:
%       problem     function (cost/grad/hess)
%       in_options  options
% Output:
%       w           solution of w
%       infos       information
%
% Reference:
%       Jorge Nocedal and Stephen Wright,
%       "Numerical optimization,"
%       Springer Science & Business Media, 2006.
%
%       sub_mode    'DAMPED'
%                   Amir Beck,
%                   "Introduction to Nonlinear Optimization Theory,
%                   Algorithms, and Applications with MATLAB,"
%                   MOS-SIAM Seris on Optimization, 2014.
%
%                   Algorithm in Section 5.2.
%
%       sub_mode    'CHOLESKY'
%                   Amir Beck,
%                   "Introduction to Nonlinear Optimization Theory,
%                   Algorithms, and Applications with MATLAB,"
%                   MOS-SIAM Seris on Optimization, 2014.
%
%                   Algorithm in Section 5.3.
%
% This file is part of GDLibrary and SGDLibrary.
%
% Created by H.Kasai on Feb. 15, 2016
% Modified by H.Kasai on Mar. 25, 2018
% Modified by H.Kasai on Oct. 20, 2020


    % set dimensions and samples
    dim = problem.dim();
    n = problem.samples(); 
    
    % set local options 
    local_options = [];    
    local_options.step_alg = 'non-backtracking'; 
    local_options.sub_mode = 'INEXACT';

    % merge options
    options = mergeOptions(get_default_options(dim), local_options);   
    options = mergeOptions(options, in_options);  


    
    % initialize
    iter = 0;  
    grad_calc_count = 0;
    w = options.w_init;
    epoch = 0;
    
    if (~strcmp(options.sub_mode, 'STANDARD')) && (~strcmp(options.sub_mode, 'CHOLESKY')) && (~strcmp(options.sub_mode, 'INEXACT')) 
        options.sub_mode = 'INEXACT';
    end
        
    % store first infos
    clear infos;    
    [infos, f_val, optgap, grad, gnorm] = store_infos(problem, w, options, [], iter, grad_calc_count, 0);
    
    % calculate hessian
   % hess = problem.full_hess(w);
    % calcualte direction
    
    d = grad;
  
    % display infos
    if options.verbose
        fprintf('Newton (%s,%s): Iter = %03d, cost = %.16e, gnorm = %.4e, optgap = %.4e\n', options.sub_mode, options.step_alg, iter, f_val, gnorm, optgap);
    end      

    % set start time
    start_time = tic();  
    max_iter = options.max_epoch;

    % main loop
   while (iter < max_iter) 
%        while (optgap > options.tol_optgap) && (gnorm > options.tol_gnorm) && (iter < options.max_epoch)     
        %rng(iter+1)
        sv = sort(randperm(dim,s));
        S = eye(dim);
        S = sqrt(dim/s)*S(:,sv);
        %S = randn(dim,s);
        %S = orth(S);
        

        if strcmp(options.step_alg, 'backtracking')
            rho = 1/2;
            c = 1e-4;
            step = backtracking_line_search(problem, -d, w, rho, c);
        elseif strcmp(options.step_alg, 'tfocs_backtracking') 
            if iter > 0
                alpha = 1.05;
                beta = 0.5; 
                step = tfocs_backtracking_search(step, w, w_old, grad, grad_old, alpha, beta);
            else
                step = options.step_init;
            end 
        else
            % do nothing
            step = options.step_init;
        end    

        % update w
        w_old = w; 
        if iter ==0
            w = w - step * d; 
            grad_old = d;
        else
            w = w - (step)*ds;
            grad_old = ds;
        end
        % proximal operator
        if ismethod(problem, 'prox')
            w = problem.prox(w, step);
        end          
        
              
        % calculate gradient
        g_old = grad;
        %grad = problem.full_grad(w);
        [F,grad] = problem.Fisher(w,S);
        g_new=grad;
        gs = S'*grad;
        ds = S*pinv(F +(norm(gs)^2)*eye(s))*gs; 
        d = ds;
        % calcualte direction  
%         if strcmp(options.sub_mode, 'STANDARD')
%              d = hess \ grad; 
%         elseif strcmp(options.sub_mode, 'CHOLESKY')
%             [L, p] = chol(hess, 'lower');
%             if p==0
%                 d = L' \ ( L \ grad);
%             else
%                 d = grad;
%             end
%         elseif strcmp(options.sub_mode, 'INEXACT')
%             [d, ~] = pcg(hess, grad, 1e-6, 1000);    
%         else
%         end

    %    d = hess*grad;
        % measure elapsed time
        elapsed_time = toc(start_time);  
        
        % count gradient evaluations
        grad_calc_count = grad_calc_count + n;  
        
        % update iter        
        iter = iter + 1;        
        epoch = epoch + 1;
        % store infos
        [infos, f_val, optgap, grad, gnorm] = store_infos(problem, w, options, infos, iter, grad_calc_count, elapsed_time);           
       
        % display infos
        if options.verbose
            fprintf('RNG(%s,%s): Iter = %03d, cost = %.16e, gnorm = %.4e, optgap = %.4e\n', options.sub_mode, options.step_alg, iter, f_val, gnorm, optgap);
        end        
    end
    
    if gnorm < options.tol_gnorm
        fprintf('Gradient norm tolerance reached: tol_gnorm = %g\n', options.tol_gnorm);
    elseif optgap < options.tol_optgap
        fprintf('Optimality gap tolerance reached: tol_optgap = %g\n', options.tol_optgap);        
    elseif iter == options.max_epoch
        fprintf('Max iter reached: max_iter = %g\n', options.max_epoch);
    end    
    
end
