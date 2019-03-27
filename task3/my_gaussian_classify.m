function[Cpreds, Ms, Covs] = my_gaussian_classify(Xtrn, Ctrn, Xtst, epsilon)
                                                   % M*D  M*1   N*D   
    [Ms,Covs] = MsandCovsConstrction(Xtrn, Ctrn, epsilon);                 % calculate mean and covariance 
   
    [Cpreds] = myGaussianSelection(Ms,Covs,Xtst,Ctrn);                          % class the probability that each class each sample belongs to
   
    [~, idx] = (max(Cpreds,[],2));                                         % keep the max probability
    Cpreds = idx;
end

function y = logdet(A)                                                     % this is used to avoid the situation logdet is NaN

U = chol(A);
y = 2*sum(log(diag(U)));

end

function [Ms,Covs] = MsandCovsConstrction(Xtrn, Ctrn, epsilon)  
    
    D = size(Xtrn,2);
    
    classes = unique(Ctrn);  
    K =  size(classes,1);   
    
    Ms = zeros(D,K);
    Covs = zeros(D,D,K);
   
    for k = 1:K                                                            % for each class k
        matrixTemperory = Xtrn(Ctrn == k,:);                               % e excerpt all the samples belongs to this class
        meanofclassk = sum(matrixTemperory,1)/size(matrixTemperory,1);     % calculate the mean value of each dimension
        Ms(:,k) = meanofclassk';                                           % put the result into Ms
        
        varianceofaclass = getCov(matrixTemperory,meanofclassk);           % call the getCov function to obtain variance
        Covs(:,:,k) =  varianceofaclass ;                                  % put this variance in to the Covs
        Covs(:,:,k) = Covs(:,:,k)+ epsilon .*eye(D,D);                     % add a small number into diagonal matrix
    end
end

function [varianceofaclass] = getCov(matrixTemperory,meanofclassk)
    zeromeanmatrix = bsxfun(@minus, matrixTemperory ,meanofclassk);        % substract mean frm each element
    amount = size(matrixTemperory,1);                                      % obtain te size of samples
    varianceofaclass = 1/(amount-1)*(zeromeanmatrix' * zeromeanmatrix);    % product of the transpose and the original matrix 
end                                                                        % and divide by (amount-1) to get variance


function [Cpreds] = myGaussianSelection(Ms,Covs,Xtst,Ctrn)
    N = size(Xtst,1);   
    
    classes = unique(Ctrn);  
    K =  size(classes,1);  
    
    Cpreds = zeros(N,K);
    
    for k = 1:K  
        mu = Ms(:,k)';
        covar = Covs(:,:,k);
        X = Xtst;
        X = X - (ones(N, 1)*mu);
        y = -(1/2)*logdet(covar);
        y = -(0.5)* sum(((X/covar).*X), 2) - (1/2)*logdet(covar);% this is the log probability presented in the lecture
        
        Cpreds(:,k) = y;
    end
end


