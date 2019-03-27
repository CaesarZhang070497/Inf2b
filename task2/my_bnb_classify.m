function [Cpreds] = my_bnb_classify(Xtrn, Ctrn, Xtst, threshold)
  
    classes = unique(Ctrn);                                                % this is the number of classes
   
    D = size(Xtrn,2);                                                      % this is the dimension of each sample
    binarisedXtrn = Xtrn>=threshold;                                       % set the element above 1 to be one and 
    binarisedXtst = Xtst>=threshold;                                       % those less then one to be zero.
    
   
    decisionMatrix = createDecisionmatrix(binarisedXtrn,Ctrn,classes,Xtrn);% form a decision matrix 
    [Cpreds] = bnbSelection(binarisedXtst,decisionMatrix,Xtst,classes);    % select the highest gaussian probability
end    
    
    
    
    
function [decisionMatrix] = createDecisionmatrix(binarisedXtrn,Ctrn,classes,Xtrn)
    D = size(Xtrn,2);
    decisionMatrix = zeros(size(classes,1),D);
    for letter = 1:size(classes,1);                                        % for each class k
            subMatrix = binarisedXtrn((Ctrn ==letter),:) ;                 % excerpt all the training sample  
            subVector = Ctrn(Ctrn ==letter);                               % and lables from that class
            denominator = size(subVector,1);                               % record the size of samples from that class
            decisionMatrix(letter,:) =  1.0e-1000 + sum(subMatrix,1)/(denominator);% calculate the probability of the occurance from that feature
    end
end


function [Cpreds] = bnbSelection(binarisedXtst,decisionMatrix,Xtst,classes)
    N = size(Xtst,1);
    Cpreds = zeros(N,size(classes,1));
    
    for n = 1:N
        supposedcandidate = binarisedXtst(n,:);                            % for each sample in the test class
        component1 =   bsxfun  (@times,decisionMatrix,supposedcandidate) ; % calculate the probability of occurance of each feature
        component2 = bsxfun(@times,1-decisionMatrix,1-binarisedXtst(n,:));
        overall = component1 + component2;                                  
        probability = log(prod((overall) ,2));                             % calculate the overall probability of esch sample belongs to each class
        Cpreds(n,:) = ((probability)');                                    % plug the probability into Cpreds
    end
    
    [~, idx] = (max(Cpreds,[],2));                                         % only keep the class with maximum probability and discard the rest
    Cpreds = idx;
end    
 

