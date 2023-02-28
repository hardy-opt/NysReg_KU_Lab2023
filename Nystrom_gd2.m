function [w, infos] = Nystrom_gd2(problem, in_options,reg,C)

    
    % set dimensions and samples
    d = problem.dim();
    n = problem.samples();
    

    % set local options 
    %local_options.sub_mode = 'Nystrom_sgd';  
    local_options.mem_size = 20;    
    
    % merge options
    options = mergeOptions(get_default_options(d), local_options);   
    options = mergeOptions(options, in_options);      
    
    
    col = options.column;
    K = options.clusters;
    P = options.partitions;
    
    %%%%%%%%%%%%%%%%
    lambda = [];
    g_nrm = [];
    dp=0;
    %%%%%%%%%%%%%%%%
    
    
    if dp==1
     %[dppX,dppy,dpp_idxp_s,dpp_idxn_s] = %DPP(X',y',P,K,plott); P=partitions, K=clusters 
        [dppX,dppy,dpp_idxp_s,dpp_idxn_s] = DPP(problem.x_train',problem.y_train',P,K,0);
    end


    % set paramters
    if options.batch_size > n
        options.batch_size = n;
    end   
    
    if ~isfield(in_options, 'batch_hess_size')
        options.batch_hess_size = 20 * options.batch_size;
    end    

    if options.batch_hess_size > n
        options.batch_hess_size = n;
    end    
        
    % initialize
    total_iter = 0;
    epoch = 0;
    grad_calc_count = 0;
    w = options.w_init;
    g = norm(problem.full_grad(w));
    num_of_bachces = floor(n / options.batch_size);     


    % store first infos
    clear infos;    
    [infos, f_val, optgap] = store_infos(problem, w, options, [], epoch, grad_calc_count, 0);  
    
    % set start time
    start_time = tic();
    
    % display infos
    if options.verbose > 0
        fprintf('%s: Epoch = %03d, cost = %.16e, optgap = %.4e\n', 'Nystrom_sgd', epoch, f_val, optgap);
    end     

    % main loop
    %while (optgap > options.tol_optgap) && (epoch < options.max_epoch)
    while (epoch < options.max_epoch)
            w0 = w;
                    set = randperm(d,col);
                   
                    if any(isnan(f_val)) || any(isinf(f_val)) || any(isnan(w0)) || any(isinf(w0))
                    return;
                    end
                    
                    if dp==1 
                    dppi=mod(epoch,P)+1;
                   % Compute Z from C and M
                   % fprintf('Size of G = %d\n',length(G)); [dpp_idxp_s{dppi};dpp_idxn_s{dppi}]
                    [Z,fn1,apta] = problem.app_hess(w0,[dpp_idxp_s{dppi};dpp_idxn_s{dppi}],set,0);
                   
                    else
                    [Z,fn1,apta] = problem.app_hess(w0,1:n,set,0);
                    end

                        lk = length(set); % k: colu
                        
                        rho = max(C*(g^2),1e-4);
                           
                        nfg = 1/rho;
                        
                        Ey = eye(lk);
                        Q = Z/(Ey+nfg*(Z'*Z));

          %  step = options.stepsizefun(total_iter, options);    
            
            % calculate gradient 
            ro = 1/2;
            cc = 1e-4;
            grad = problem.full_grad(w);

                     %NI =  nfg*eye(d) - (nfg)*Z/(Ey+nfg*(Z'*Z))*Z';
                     vect = Z'*grad;
                    
                     NI = nfg*(grad - nfg*(Q*vect));
                     g = norm(grad);
                     step = backtracking_line_search(problem, -NI, w, ro, cc);
                     %steps = max(step,1e-3);
                     v = step*NI;
                     w = w - v;
            
            % proximal operator
            if ismethod(problem, 'prox')
                w = problem.prox(w, step);
            end              
                     
            
            total_iter = total_iter + 1;
        
        
        
        % measure elapsed time
        elapsed_time = toc(start_time);
        
        lambda = [lambda 1/nfg];
       % grad = problem.grad(w,1:n);
         g_nrm = [g_nrm g];
        
        % count gradient evaluations
        grad_calc_count = grad_calc_count + epoch * options.batch_size;        
        epoch = epoch + 1;

        
        % store infos
        [infos, f_val, optgap] = store_infos(problem, w, options, infos, epoch, grad_calc_count, elapsed_time);            

        % display infos
        if options.verbose > 0 %&& mod(epoch,10)==1
            fprintf('%s: Ep = %3d, cost = %.5e, optgap = %.4e, time=%0.3f, |g| = %.4e, rho = %.4e, del = %s\n', 'Nystrom_gd2', epoch, f_val, optgap,elapsed_time,g,1/nfg,C);
        end
    end
    
    if optgap < options.tol_optgap
        fprintf('Optimality gap tolerance reached: tol_optgap = %g\n', options.tol_optgap);
    elseif epoch == options.max_epoch
        fprintf('Max epoch reached: max_epochr = %g\n', options.max_epoch);
    end
      
%  infos.RHO = RH;
infos.lambda = lambda';
infos.g_nrm = g_nrm';
end

