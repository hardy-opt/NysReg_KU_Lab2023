function [w, infos] = Nystrom_gdLM2(problem, in_options,reg,dp,C,D)
%http://www.amp.i.kyoto-u.ac.jp/tecrep/index-e.html
    %rho is replaced by reg+rho on 12th Jan 2022.
    % If dp = 1 then NSGD-DP
    % else NSGD


    % set dimensions and samples
    d = problem.dim();
    n = problem.samples();
    

    % set local options 
    local_options.sub_mode = 'Nystrom_sgd';  
    local_options.mem_size = 20;    
    
    % merge options
    options = mergeOptions(get_default_options(d), local_options);   
    options = mergeOptions(options, in_options);      
    
    
    col = options.column;
    K = options.clusters;
    P = options.partitions;
    
    
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
    
    RH =[];  rh_old = 0; 
    
    % initialize
    total_iter = 0;
    epoch = 0;
    grad_calc_count = 0;
    w = options.w_init;
    grad = problem.grad(w,1:n);
    g = norm(grad);
    num_of_bachces = floor(n / options.batch_size);     


    % store first infos
    clear infos;    
    [infos, f_val, optgap] = store_infos(problem, w, options, [], epoch, grad_calc_count, 0);  
    
    % set start time
    start_time = tic();
    
    % display infos
    if options.verbose > 0
        fprintf('%s: Epoch = %03d, cost = %.16e, optgap = %.4e\n', 'Nystrom_sgdLM2', epoch, f_val, optgap);
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

                        lk = length(set); % k: colums
                        % HI = inv(H); % Hessian Inverse
                        delta = D;
                        
                        rho = max(C*(g^2),reg);
                        nfg = 1/(rho);
                        Ey = eye(lk);
                        ZZ  = Z'*Z;
                        Q = (nfg^2)*(Z/(Ey+nfg*(ZZ)*(ZZ)));
                        
            % update step-size
           % step = options.stepsizefun(total_iter, options);
            ro = 1/2;
            cc = 1e-4;
            
            % calculate gradient
            grad = problem.full_grad(w);
            g = norm(grad);
            
                     %NI =  nfg*eye(d) - (nfg)*Z/(Ey+nfg*(Z'*Z))*Z';
                     p1 = delta*grad;
                     p2 = Z'*grad;
                     v1 = (Z*p2)+p1;        
                     v2 = Z'*v1;
                     v3 = ZZ*v2;
                     NI = nfg*v1 - (Q*v3); 
                     step = backtracking_line_search(problem, -NI, w, ro, cc);
                     v = step*NI;
                     w = w - v;
            
            % proximal operator
            if ismethod(problem, 'prox')
                w = problem.prox(w, step);
            end              
                     
            
            total_iter = total_iter + 1;
       
        
             
        %vr = norm(step*v-step*problem.grad(w,1:n))^2;

        % measure elapsed time
        elapsed_time = toc(start_time);
        
        % count gradient evaluations
        grad_calc_count = grad_calc_count + epoch * options.batch_size;        
        epoch = epoch + 1;

        
        % store infos
        [infos, f_val, optgap] = store_infos(problem, w, options, infos, epoch, grad_calc_count, elapsed_time);            

        % display infos
        if options.verbose > 0
            fprintf('%s: Epoch = %03d, cost = %.6e, optgap = %.4e, time=%0.3f\n', 'Nystrom_gdLM2', epoch, f_val, optgap,elapsed_time);
        end
    end
    
    if optgap < options.tol_optgap
        fprintf('Optimality gap tolerance reached: tol_optgap = %g\n', options.tol_optgap);
    elseif epoch == options.max_epoch
        fprintf('Max epoch reached: max_epochr = %g\n', options.max_epoch);
    end
      
  infos.delta = C;

end

