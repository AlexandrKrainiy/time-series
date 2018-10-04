function PSORF(target)
Len=size(target,1);
len1=round(Len*0.8);
traindata=target(1:len1);

nVar=2;            % Number of Decision Variables

VarSize=[1 nVar];   % Size of Decision Variables Matrix

VarMin=-10;         % Lower Bound of Variables
VarMax= 10;         % Upper Bound of Variables


%% PSO Parameters

MaxIt=100;      % Maximum Number of Iterations

nPop=50;        % Population Size (Swarm Size)

% PSO Parameters
w=1;            % Inertia Weight
wdamp=0.99;     % Inertia Weight Damping Ratio
c1=1.5;         % Personal Learning Coefficient
c2=2.0;         % Global Learning Coefficient


% Velocity Limits
VelMax=0.1*(VarMax-VarMin);
VelMin=-VelMax;

%% Initialization

empty_particle.Position=[];
empty_particle.Cost=[];
empty_particle.Velocity=[];
empty_particle.Best.Position=[];
empty_particle.Best.Cost=[];

particle=repmat(empty_particle,nPop,1);

GlobalBest.Cost=inf;

for i=1:nPop
    
    % Initialize Position
    particle(i).Position=unifrnd(VarMin,VarMax,VarSize);
    
    % Initialize Velocity
    particle(i).Velocity=zeros(VarSize);
    
    % Evaluation
    particle(i).Cost=CostFunction(traindata,particle(i).Position);
    
    % Update Personal Best
    particle(i).Best.Position=particle(i).Position;
    particle(i).Best.Cost=particle(i).Cost;
    
    % Update Global Best
    if particle(i).Best.Cost<GlobalBest.Cost
        
        GlobalBest=particle(i).Best;
        
    end
    
end

BestCost=zeros(MaxIt,1);

%% PSO Main Loop

for it=1:MaxIt
    
    for i=1:nPop
        
        % Update Velocity
        particle(i).Velocity = w*particle(i).Velocity ...
            +c1*rand(VarSize).*(particle(i).Best.Position-particle(i).Position) ...
            +c2*rand(VarSize).*(GlobalBest.Position-particle(i).Position);
        
        % Apply Velocity Limits
        particle(i).Velocity = max(particle(i).Velocity,VelMin);
        particle(i).Velocity = min(particle(i).Velocity,VelMax);
        
        % Update Position
        particle(i).Position = particle(i).Position + particle(i).Velocity;
        
        % Velocity Mirror Effect
        IsOutside=(particle(i).Position<VarMin | particle(i).Position>VarMax);
        particle(i).Velocity(IsOutside)=-particle(i).Velocity(IsOutside);
        
        % Apply Position Limits
        particle(i).Position = max(particle(i).Position,VarMin);
        particle(i).Position = min(particle(i).Position,VarMax);
        
        % Evaluation
        particle(i).Cost = CostFunction(traindata,particle(i).Position);
        
        % Update Personal Best
        if particle(i).Cost<particle(i).Best.Cost
            
            particle(i).Best.Position=particle(i).Position;
            particle(i).Best.Cost=particle(i).Cost;
            
            % Update Global Best
            if particle(i).Best.Cost<GlobalBest.Cost
                
                GlobalBest=particle(i).Best;
                
            end
            
        end
        
    end
    
    BestCost(it)=GlobalBest.Cost;
    
%     disp(['Iteration ' num2str(it) ': Best Cost = ' num2str(BestCost(it))]);
    
    w=w*wdamp;
    
end

BestSol = GlobalBest;
predictdata=predict_pso(target, BestSol.Position,Len-len1,len1);
nTrees = 20;
 features =target(1:len1-1);
classLabels =target(2:len1);
% Train the TreeBagger (Decision Forest).
B = TreeBagger(nTrees,features,classLabels, 'Method', 'classification');
 testpredict= B.predict(predictdata);
 predictdata=zeros(size(testpredict,1),1);
 for i=1:size(testpredict,1)
     predictdata(i)=str2double(cell2mat(testpredict(i)));
 end
 mape=100/(Len-len1)*sum(abs(target(len1+1:Len)-predictdata)./target(len1+1:Len));
 rmse=1/(Len-len1)*sqrt((target(len1+1:Len)-predictdata)'*(target(len1+1:Len)-predictdata));
 figure
     plot(1:Len,target,'b');
     hold on;
     plot(len1+1:Len,predictdata,'r');
     title(['PSORF model(mape=', num2str(mape),', rmse=',num2str(rmse),')']);  
     legend( 'Actual Future Price','Predicted Price');
end
function z=CostFunction(adata, coe)
coe=coe';
n=size(adata,1);
m=size(coe,1);
tdata=zeros(n-m+1,1);
for i=1:size(tdata,1)
    tdata(i)=adata(i:i+m-2,1)'*coe(1:m-1,1)+coe(m);
end
z=(tdata-adata(m:n,1))'*(tdata-adata(m:n,1))/(n-m+1);
end
function z=predict_pso(adata, coe,num,no)
coe=coe';
z=zeros(num,1);
m=size(coe,1);
for i=1:num
    z(i)=adata(no-m+1+i:no+i-1,1)'*coe(1:m-1,1)+coe(m);
end
end