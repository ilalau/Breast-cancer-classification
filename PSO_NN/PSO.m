function [x, err] = PSO(CostFunction, nVar)
% Cost Function - MSE criterion
% nVar = Number of Decision Variables.

VarSize = [1 nVar]; % Size of Decision Variables Matrix

VarMin = -7;        % Lower Bound of Variables
VarMax = 7;         % Upper Bound of Variables

%% PSO Parameters
MaxIter = 100;       % Maximum Number of Iterations
nPop = 50;          % Population size (Swarm Size)

% PSO Parameters
w = 0.7;              % Inertia Weight
c1 = 2.0;           % Personal Learning Coefficient
c2 = 2.0;           % Global Learning Coefficient

% Velocity Limits
VelMax = 0.1*(VarMax - VarMin);
VelMin = -VelMax;

%% Initializtion
init_particle.Position = [];
init_particle.Cost = [];
init_particle.Velocity = [];
init_particle.Best.Position = [];
init_particle.Best.Cost = [];

particle = repmat(init_particle, nPop, 1);

GlobalBest.Cost = inf;

for i = 1:nPop
    % Initialize Position
    particle(i).Position = unifrnd(VarMin, VarMax, VarSize);
    
    % Initialize Velocity
    particle(i).Velocity = zeros(VarSize);
    
    % Evaluation
    particle(i).Cost = CostFunction(particle(i).Position);
    
    % Update Personal Best
    particle(i).Best.Position = particle(i).Position;
    particle(i).Best.Cost = particle(i).Cost;
    
    % Update Global best
    if particle(i).Best.Cost < GlobalBest.Cost
        GlobalBest = particle(i).Best;
    end
end

BestCost = zeros(MaxIter, 1);

%% PSO Algorithm
for it = 1:MaxIter
    for i = 1:nPop
        % Update Velocity
        particle(i).Velocity = w*particle(i).Velocity ...
            + c1*rand(VarSize).*(particle(i).Best.Position - particle(i).Position) ...
            + c2*rand(VarSize).*(GlobalBest.Position - particle(i).Position);
        
        % Apply Velocity Limits
        particle(i).Velocity = max(particle(i).Velocity, VelMin);
        particle(i).Velocity = min(particle(i).Velocity, VelMax);
        
        % Update Position
        particle(i).Position = particle(i).Position + particle(i).Velocity;
        
        % Velocity Mirror Effect
        IsOutside=(particle(i).Position < VarMin | particle(i).Position > VarMax);
        particle(i).Velocity(IsOutside) = -particle(i).Velocity(IsOutside);
        
        % Apply Position Limits
        particle(i).Position = max(particle(i).Position, VarMin);
        particle(i).Position = min(particle(i).Position, VarMax);
        
        % Evaluation
        particle(i).Cost = CostFunction(particle(i).Position);
        
        % Update Personal Best
        if particle(i).Cost < particle(i).Best.Cost
            particle(i).Best.Position = particle(i).Position;
            particle(i).Best.Cost = particle(i).Cost;
            
            % Update Globale Best
            if particle(i).Best.Cost < GlobalBest.Cost
                GlobalBest = particle(i).Best;
            end
        end
        
    end
    
    BestCost(it) = GlobalBest.Cost;
    
    disp(['Iteration ' num2str(it) ': Best Cost = ' num2str(BestCost(it))]);
    
end
   
BestSol = GlobalBest;
x = BestSol.Position';
err = BestSol.Cost;

%% Results
figure;
semilogy(BestCost, 'LineWidth', 2);
xlabel('Iteration');
ylabel('Best Cost');
grid on;

