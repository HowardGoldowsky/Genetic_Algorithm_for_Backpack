
% Genetic algorithm to solve the backpack problem.

% A backpack can hold maxWeight weight. Each item has an importance. The 
% goal is to stuff the backpack with the most valuable payload. 

function solution = backpackGA_01(maxNumGenerations, numHypotheses, cullRate, mutationRate, initRate, maxWeight, objects)

% INITIALIZATIONS

load(objects);
numGen = 0;
numObjects = numel(objects);
genome = rand(numHypotheses, numObjects) < initRate;                        % init random sets of objects (1/0 = in/not in backpack)
numBest = floor(cullRate*numHypotheses);                                    % number of best hypotheses from which to select reproducers

while (numGen < maxNumGenerations)                                          % MAIN LOOP
    
    % CULL GENOME
    
    [fitness, idxFitness] = ComputeFitness(genome, objects, maxWeight);     % returns fitness and rank of each hypothesis in genome
    genome = genome(idxFitness(1:numBest),:);                               % cull the best hypotheses for reproduction          
    totalFitness = sum(fitness(idxFitness(1:numBest)));                     % total fitness of the culled genome
    probabilities = fitness(idxFitness(1:numBest))/totalFitness;            % probabilities of culled genome relative to fitness
    
    % SAMPLE HYPOTHESES TO PREPARE A NEW GENERATION FOR CROSSOVER
    
    trials = rand(numHypotheses,1);                                         % random trials
    PMF = cumsum(probabilities);                                            % probability mass function based on probabilities of culled hypotheses
    SelectHypotheses = @(r) find(r < PMF, 1, 'first');                      % anonymous function that finds the 1st index of each trial < PMF 
    idxGenome = arrayfun(SelectHypotheses,trials);                          % proportional to fitness, randomly select indexes of genome to crossover
    idxCrossoverLocation = randi([1,numObjects-1],numHypotheses/2,1);       % random crossover location
    
    % CROSSOVER
    
    genome = PerformCrossover(genome, idxGenome, idxCrossoverLocation);
    
    % MUTATION

    mutationMap = rand(cullRate*numHypotheses, numObjects) < mutationRate;  % element will equal true if rand < mutationRate
    genome = xor(genome, mutationMap);                                      % swap genome(bit) if mutationMap(bit)==true

    numGen = numGen + 1;
    
end % while

[~, idxFitness] = ComputeFitness(genome, objects, maxWeight);               % rank genome one last time and select the top performer

solution.genome         = genome(idxFitness(1),:);
solution.fitness        = sum([objects.importance].*genome(idxFitness(1),:));
solution.total_weight   = sum([objects.weight].*genome(idxFitness(1),:));
solution.importance     = [objects.importance].*genome(idxFitness(1),:);
solution.weight         = [objects.weight].*genome(idxFitness(1),:);
    
end % function

function [fitness, idxFitness] = ComputeFitness(genome, objects, maxWeight)

    weights     = repmat([objects.weight],length(genome),1);                % data structure to represent weights of objects
    importances = repmat([objects.importance],length(genome),1);            % data structure to represent importances of objects
    
    fitness = sum(importances.*genome,2);                                   % fitness of each hypothesis
    overweight = sum(weights.*genome,2) > maxWeight;                        % Bool vector representing which hypothesis are overweight
    fitness(overweight) = -fitness(overweight);                             % penalize overweight backpacks proportionally to their total weight
    [~, idxFitness] = sort(fitness,'descend');

end % function ComputeFitness()

function genome = PerformCrossover(genome, idxGenome, idxCrossoverLocation)

    for hyp = 1:numel(idxCrossoverLocation)
        
        idxA = idxGenome(2*hyp-1);                                          % pointer to the ith selected hypothesis in the genome
        idxB = idxGenome(2*hyp);                                            % pointer to the ith+1 selected hypothesis in the genome
        genome(idxA,:) = [genome(idxB,1:idxCrossoverLocation(hyp)),genome(idxA,idxCrossoverLocation(hyp)+1:end)];
        genome(idxB,:) = [genome(idxA,1:idxCrossoverLocation(hyp)),genome(idxB,idxCrossoverLocation(hyp)+1:end)];
        
    end

end % function PerformCrossover()
