function [Cpreds] = my_knn_classify(Xtrn, Ctrn, Xtst, Ks)
    dicisionMatrix = createDecisionmatrix(Xtrn, Xtst);                     % call the function 'createDecisionmatrix' to work out the decisionmatrix
    
    Cpreds = knnSelection(dicisionMatrix,Xtst,Ctrn,Ks);                    % call the function 'knnSelection' to work out the matrix of test set's label 
end

function [dicisionMatrix] = createDecisionmatrix(Xtrn, Xtst)               % this function is the implementation of the algorithm given in FAQ section
    M = size(Xtrn,1);
    N = size(Xtst,1);
    XtrnXtrn = sum(bsxfun(@power,Xtrn,2),2); 
    XtstXtst = sum(bsxfun(@power,Xtst,2),2);  
    
    extendedxx = repmat(XtstXtst,1,M);
    extendedyy = ((repmat(XtrnXtrn,1,N))');
    
    overall = bsxfun(@plus,extendedxx,extendedyy);
    dicisionMatrix = bsxfun(@minus,overall,2.* Xtst*(Xtrn'));
end

function [Cpreds] = knnSelection(dicisionMatrix,Xtst,Ctrn,Ks)
    N = size(Xtst,1);
    L = size(Ks,1);
    Cpreds = zeros(N,L);
    [~ , sortedindex] = sort(dicisionMatrix,2,'ascend');                   % sorted the decision matrix along its rows and obtained a matrix of sorted index.
    
    for l = 1:L                                                            % segment the index matrix 
    a = sortedindex(:,1:Ks(l));                                            % according to the desired k neighbourhood
    b = Ctrn(a);
    modematrix = mode(b,2);                                                % take the class of the mode occurance of the k class
    
    classlabel = modematrix(:,1);
    Cpreds(:,l) = classlabel;
    end 
end