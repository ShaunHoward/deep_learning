require 'torch'
require 'nn'
require 'optim'

-- import handwritten digits data
mnist = require 'mnist'

-- import paths for saving torch model
paths = require 'paths'

-- import training and test data
fullset = mnist.traindataset()
testset = mnist.testdataset()

-- allocate training and validation data from full data set
trainset = {size = 50000,
            data = fullset.data[{{1,50000}}]:double(),
            label = fullset.label[{{1,50000}}]}

validationset = {size = 10000,
                 data = fullset.data[{{50001, 60000}}]:double(),
                 label = fullset.label[{{50001, 60000}}]}

-- create sequential NN model
model = nn.Sequential()

-- create layers of NN in sequence
model:add(nn.Reshape(28*28))
model:add(nn.Linear(28*28, 30))
model:add(nn.Tanh())
model:add(nn.Linear(30, 10))
model:add(nn.LogSoftMax())

-- NLL criterion requires the NN output to have log-probabilities of each class
-- LogSoftMax is required above for this reason
criterion = nn.ClassNLLCriterion()

-- create stochastic gradient descent parameters
sgd_params = {learningRate = 1e-2,
              learningRateDecay = 1e-4,
              weightDecay = 1e-3,
              momentum = 1e-4}

x, dl_dx = model:getParameters()

-- create step function for a single epoch of training and returns loss value
step = function(batch_size)
	local current_loss = 0
	    local count = 0
	    local shuffle = torch.randperm(trainset.size)
	    batch_size = batch_size or 200
	    
	    for t = 1,trainset.size,batch_size do
	        -- setup inputs and targets for this mini-batch
	        local size = math.min(t + batch_size - 1, trainset.size) - t
	        local inputs = torch.Tensor(size, 28, 28)
	        local targets = torch.Tensor(size)
	        for i = 1,size do
	            local input = trainset.data[shuffle[i+t]]
	            local target = trainset.label[shuffle[i+t]]
	            -- if target == 0 then target = 10 end
	            inputs[i] = input
	            targets[i] = target
	        end
	        targets:add(1)
	        
	        local feval = function(x_new)
	            -- reset data
	            if x ~= x_new then x:copy(x_new) end
	            dl_dx:zero()

	            -- perform mini-batch gradient descent
	            local loss = criterion:forward(model:forward(inputs), targets)
	            model:backward(inputs, criterion:backward(model.output, targets))

	            return loss, dl_dx
	        end
	        
	        _, fs = optim.sgd(feval, x, sgd_params)
	        -- fs is a table containing value of the loss function
	        -- (just 1 value for the SGD optimization)
	        count = count + 1
	        current_loss = current_loss + fs[1]
	    end

	    -- normalize loss
	    return current_loss / count
	end

-- create accuracy evaluation function
eval = function(dataset, batch_size)
    local count = 0
    batch_size = batch_size or 200
    
    for i = 1,dataset.size,batch_size do
        local size = math.min(i + batch_size - 1, dataset.size) - i
        local inputs = dataset.data[{{i,i+size-1}}]
        local targets = dataset.label[{{i,i+size-1}}]:long()
        local outputs = model:forward(inputs)
        local _, indices = torch.max(outputs, 2)
        indices:add(-1)
        local guessed_right = indices:eq(targets):sum()
        count = count + guessed_right
    end

    return count / dataset.size
end

-- set up training code
MAX_ITERS = 30
do
    local last_accuracy = 0
    local decreasing = 0
    local threshold = 1 -- how many deacreasing epochs we allow
    for i = 1,MAX_ITERS do
        local loss = step()
        print(string.format('Epoch: %d Current loss: %4f', i, loss))
        local accuracy = eval(validationset)
        print(string.format('Accuracy on the validation set: %4f', accuracy))
        if accuracy < last_accuracy then
            if decreasing > threshold then break end
            decreasing = decreasing + 1
        else
            decreasing = 0
        end
        last_accuracy = accuracy
    end
end

-- test the model accuracy
testset.data = testset.data:double()
eval(testset)

-- open file and save torch model
filename = paths.concat(paths.cwd(), 'model.net')
torch.save(filename, model)

-- restore model with this code
loaded_model = torch.load(filename)

-- redefine evaluation function using loaded model
loaded_eval = function(dataset)
   local count = 0
   for i = 1,dataset.size do
      local output = model1:forward(dataset.data[i])
      local _, index = torch.max(output, 1) -- max index
      local digit = index[1] % 10
      if digit == dataset.label[i] then count = count + 1 end
   end

   return count / dataset.size
end

-- re-evaluate testset
loaded_eval(testset)
