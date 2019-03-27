

function [Cpreds] = my_improved_gaussian_classify(Xtrn, Ctrn, Xtst)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%PCA transformation
    [coeff,scoreXtrn] = myPCA(Xtrn);                                       % this function is my implementation pf pca()
  
    %  Xtst = bsxfun(@minus,Xtst,mean(Xtst));
    Xtrn = Xtrn*coeff(:,1:30);                                             % this is the amount of the most important dimensions 
    Xtst = Xtst* coeff(:,1:30);                                            % chosen to keep

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    M = size(Xtrn,1);
    N = size(Xtst,1);
    rearrangedCtrn = zeros(size(Ctrn));                                    % this is the variable to store new variable
    subclassamount =12;                                                    % this is the amount of subclass
    classes = unique(Ctrn);                                                % this is the number of different amount of classes
    K = size(classes,1);                                                  
    for k = 1:K                                                            % for each class k 
        matrixTemperory = Xtrn(Ctrn == k,:);                               % this matrix contains all the samples belongs to thia class
        indexVector = myKmeans(matrixTemperory,subclassamount);            % my kmeans determines which subclass a sample belongs to
        indexVector = subclassamount * (k - 1) + indexVector;              % create a unique identifier of a sample
        rearrangedCtrn(Ctrn == k) = indexVector;                           % put this sample into the designated place in variable 'rearrangedCtrn'
    end
    
    epsilon = 0.09;                                                        % set epsilon to a fixed number
    [Cpreds, ~, ~] = my_gaussian_classify(Xtrn, rearrangedCtrn, Xtst,epsilon,subclassamount);% call the classfier to classify test samples
    
   for k = 1:K
        for n = 1:N
            if Cpreds(n)>=(k-1)*subclassamount && Cpreds(n)<=subclassamount*k
                Cpreds(n) = k;                                             % convert the label to original form
            end
        end
    end
   
end

function [idx] = myKmeans(matrixTemperory,subclassamount)                  % this is my implementation of k-means
    
    number = size(matrixTemperory,1);                                      % this is the size of input 
    centers = matrixTemperory(1:round(number/subclassamount):number,:);    % choosing initial centers from initial intervals

    maxiter = 100;                                                         % dccide how many iterations 
    for i = 1:maxiter                                                      % for each iterations
   
    YY = sum(bsxfun(@power,matrixTemperory,2),2);  %m * D traning data set % this is the algorithm given in FAQ section
    XX = sum(bsxfun(@power,centers,2),2);                                  %n * D test data set
    
    extendedxx = repmat(XX,1,number);
    extendedyy = ((repmat(YY,1,size(centers,1)))');% m * D  D*N
    
    overall = bsxfun(@plus,extendedxx,extendedyy);
    decisionMatrix = bsxfun(@minus,overall,2.* centers*(matrixTemperory'));
    [~, idx] = min(decisionMatrix);
             
        for entry = 1:subclassamount
            centers(entry,:) = myMean   (  matrixTemperory(idx == entry,:));% redetermine the location of each center
            
        end
    end
end

function [coeff, score] = myPCA(X)
 
    meanX = sum(X,1)/size(X,1);
  
    X = bsxfun(@minus,X,meanX);                                            % center the values
    X_T = transpose(X);                                                    % Transpose X to calc C                       
    cov = (1/size(X,1)) * X_T * X;                                         % Use the formula gives to calulate C
    [PC, V] = eig(cov);
   
    V = diag(V);                                                           % PC are the principal components, i.e. eigenvectors
    [tmp, ridx] = sort(V, 1, 'descend');                                   % and V are the corresponding eigenvalues 
   
    PC = PC(:,ridx);                                                       % eigenvalues = D = princ. values
    coeff= PC;                                                             % Convert the matrix of principal values returned in 3 into a vector.
    score= X*PC;
end 

function [result] = myMean(matrix)                                         % this is my own implementation of mean()
    number= size(matrix,1);
    result = sum(matrix,1)/number;
end

function[Cpreds, Ms, Covs] = my_gaussian_classify(Xtrn, Ctrn, Xtst, epsilon,subclassamount)
                                                   % M*D  M*1   N*D
                                                   %  this is the
                                                   %  classification
                                                   %  function in 3.1

                  
   
    [Ms,Covs] = MsandCovsConstrction(Xtrn, Ctrn, epsilon,subclassamount);
    [Cpreds] = myGaussianSelection(Ms,Covs,Xtst,subclassamount,Ctrn);
   
    [~, idx] = (max(Cpreds,[],2));
    Cpreds = idx;
end

function y = logdet(A)

U = chol(A);
y = 2*sum(log(diag(U)));

end

function [Ms,Covs] = MsandCovsConstrction(Xtrn, Ctrn, epsilon,subclassamount)

    D = size(Xtrn,2);
    
     classes = unique(Ctrn);  
    K =  size(classes,1)/subclassamount;  
    
    Ms = zeros(D,K*subclassamount);
    Covs = zeros(D,D,K*subclassamount);


    for k = 1:K*subclassamount
        
        matrixTemperory = Xtrn(Ctrn == k,:);
        meanofclassk = sum(matrixTemperory,1)/size(matrixTemperory,1);
        
     
        
        Ms(:,k) = meanofclassk';
        zeromeanmatrix = bsxfun(@minus, matrixTemperory ,meanofclassk);
        amount = length(matrixTemperory);
        varianceofaclass = (1/(amount-1))*(zeromeanmatrix' * zeromeanmatrix);
     
        Covs(:,:,k) = varianceofaclass  ;
        Covs(:,:,k)= Covs(:,:,k) + epsilon .*eye(D,D);
        
    end
end

function [Cpreds] = myGaussianSelection(Ms,Covs,Xtst,subclassamount,Ctrn)
    N = size(Xtst,1);   
    classes = unique(Ctrn);  
    K =  size(classes,1)/subclassamount;  
    Cpreds = zeros(N,K);
    
    for k = 1:K* subclassamount 
       
        mu = Ms(:,k)';
        covar = Covs(:,:,k);
        X = Xtst;
        X = X - (ones(N, 1)*mu);
        
        y = -(0.5)* sum(((X/covar).*X), 2) - (1/2)*logdet(covar);
     
        Cpreds(:,k) = y;
    end
end


