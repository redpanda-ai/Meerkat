--[[
Data Program for Crepe
By Xiang Zhang @ New York University
--]]

require("image")
local ffi = require("ffi")

-- The class
local Data = torch.class("Data")

function Data:__init(config)
   -- Alphabet settings
   self.alphabet = config.alphabet or "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
   self.dict = {}
   for i = 1,#self.alphabet do
      self.dict[self.alphabet:sub(i,i)] = i
   end

   self.length = config.length or 1014
   self.batch_size = config.batch_size or 128
   self.file = config.file
   self.prob = config.prob
   self.padding = config.padding
   self.scale = config.scale
   self.extra = config.extra

   self.config = config
   self.data = torch.load(self.file)

   self.flattened_index = self:flattenIndex(self.data.index)
   self.total_size = self.flattened_index:size(1)

   self.all_label_numbers = self:generateAllLabels(self.data.index)

   if self.prob then
      for i = 1, #self.prob - 1 do
	 self.prob[i + 1] = self.prob[i] + self.prob[i + 1]
      end
   end
end

-- Flatten Index into one tensor
function Data:flattenIndex(index)
   local t = index[1]
   for i = 2, #index do
      t = torch.cat(t, index[i], 1)
      collectgarbage()
   end
   return t
end

-- Generate a tensor of all label numbers
function Data:generateAllLabels(index)
   local all_labels = torch.Tensor(#index[1]):fill(1)
   for i = 2, #index do
      cur_labels = torch.Tensor(#index[i]):fill(i)
      all_labels = torch.cat(all_labels, cur_labels, 1)
      collectgarbage()
   end
   return all_labels
end

function Data:nClasses()
   return #self.data.index
end

function Data:getBatch(inputs, labels, data, extra)
   local data = data or self.data
   local extra = extra or self.extra
   local inputs = inputs or torch.Tensor(self.batch_size, #self.alphabet, self.length)
   local labels = labels or torch.Tensor(inputs:size(1))
   local sampling_flag = false

   for i = 1, inputs:size(1) do
      local label, s
      local random_char_start, random_index
      if sampling_flag then
         -- Randomly select one sample from all the training data
         random_index = torch.random(self.total_size)
         random_char_start = self.flattened_index[random_index]

         -- Get the string content of the selected sample
         s = ffi.string(torch.data(data.content:narrow(1, random_char_start[1], 1)))

         -- Get the label of the selected sample
         label = self.all_label_numbers[random_index]
         labels[i] = label
      else
         -- Choose data
         label = torch.random(#data.index)
         --if math.random() > 0.85 then label = 1 end
         local input = torch.random(data.index[label]:size(1))
         s = ffi.string(torch.data(data.content:narrow(1, data.index[label][input][data.index[label][input]:size(1)], 1)))
         for l = data.index[label][input]:size(1) - 1, 1, -1 do
	        s = s.." "..ffi.string(torch.data(data.content:narrow(1, data.index[label][input][l], 1)))
         end
         labels[i] = label
      end
      -- Quantize the string
      self:stringToTensor(s, self.length, inputs:select(1, i))
      sampling_flag = not sampling_flag
   end

   return inputs, labels
end

function Data:iterator(static, data)
   local i = 1
   local j = 0
   local data = data or self.data
   local static
   if static == nil then static = true end

   if static then
      inputs = torch.Tensor(self.batch_size, #self.alphabet, self.length)
      labels = torch.Tensor(inputs:size(1))
   end

   return function()
      if data.index[i] == nil then return end

      local inputs = inputs or torch.Tensor(self.batch_size, #self.alphabet, self.length)
      local labels = labels or torch.Tensor(inputs:size(1))
      local trans = {}

      local n = 0
      for k = 1, inputs:size(1) do
	 j = j + 1
	 if j > data.index[i]:size(1) then
	    i = i + 1
	    if data.index[i] == nil then
	       break
	    end
	    j = 1
	 end
	 n = n + 1
	 local s = ffi.string(torch.data(data.content:narrow(1, data.index[i][j][data.index[i][j]:size(1)], 1)))
	 for l = data.index[i][j]:size(1) - 1, 1, -1 do
	    s = s.." "..ffi.string(torch.data(data.content:narrow(1, data.index[i][j][l], 1)))
	 end
	 local data = self:stringToTensor(s, self.length, inputs:select(1, k))
	 labels[k] = i
         trans[k] = s
      end

      return inputs, labels, n, trans
   end
end

function Data:stringToTensor(str, l, input, p)
   local s = str:lower()
   local l = l or #s
   local t = input or torch.Tensor(#self.alphabet, l)
   t:zero()
   for i = #s, math.max(#s - l + 1, 1), -1 do
      if self.dict[s:sub(i,i)] then
	 t[self.dict[s:sub(i,i)]][#s - i + 1] = 1
      end
   end
   return t
end

return Data
