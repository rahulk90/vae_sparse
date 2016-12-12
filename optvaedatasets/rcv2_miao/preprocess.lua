--Sent via email by Yishu Miao (yishu.miao@cs.ox.ac.uk)
require 'torch'

dir ='./'--./data/RCV1-v2/raw/'
des ='./'--./data/RCV1-v2/'

raw_full = dir..'train.txt'
raw_train_file = dir..'train.txt'

vocab_raw = des..'vocabulary.txt'
vocab_file = des..'vocab.new'
train_feat = des..'train.feat'
test_feat = des..'test.feat'

--------------------------------------------
local split = function(s, p)
    local rt= {}
    string.gsub(s, '[^'..p..']+', function(w) table.insert(rt, w) end )
    return rt
end
local randomSelect = function( total, n,m )
    local num_set = {}
    local map = {}
    while #num_set < total do
        local num = math.random(n,m)
        if not map[num] then
            map[num] = #num_set + 1
            num_set[#num_set+1] = num
        end
    end 
    return num_set
end
local findElement = function( tab, element )
    for _,e in pairs(tab) do
        if e == element then
            return true
        end
    end
    return false
end

--------------------------------------------
-- 0. ID:freq
Id2Word = {}
Word2ID = {}
term_freq = {}
train_label_set = {}

-- build vocab
fin = io.open(raw_full)
for line in fin:lines() do
    local words = split(line,' ')
    if words[1] == '.I' then
        train_label_set[#train_label_set+1] = tonumber(words[2])
    else
        if words[1] ~= '.W' then
            for _,word in pairs(words) do
                if not Word2ID[word] then
                    Id2Word[#Id2Word+1] = word
                    Word2ID[word] = #Id2Word
                end
                term_freq[Word2ID[word] ] = ( term_freq[Word2ID[word] ] or 0)+1
            end
        end
    end
end
fin:close()

print ('Built vocabulary')

fout = io.open(vocab_raw,'w')
for i =1,#Id2Word do
    fout:write(Id2Word[i]..'\n')
end
fout:close()

-- rank 
term_set = {}
for k, v in pairs(term_freq) do
    term_set[#term_set+1] = {key = k, val = v}
end
table.sort(term_set, function(x1,x2) return x1.val>x2.val end )

-- 1. Filter words
volume = 0
filt_map = {}       -- orginal key (0~61188) to filter key(1~ ...)
filt_freq = {}
fout = io.open(vocab_file,'w')

for i = 1,10000 do      
    if term_set[i].val >2 then     
        volume = volume+1
        filt_map[tonumber(term_set[i].key)] = volume     -- make a map
        filt_freq[#filt_freq+1] = term_set[i].val
        fout:write(Id2Word[tonumber(term_set[i].key)]..' '..term_set[i].val..'\n')        
    end
end
fout:close()


--2. build set 
local idx = 0
train_set = {}

local isTest = randomSelect(10000,1,804414)

fin = io.open(raw_train_file)
fout_train = io.open(train_feat,'w')
fout_test = io.open(test_feat,'w')

for line in fin:lines() do
    words = split(line,' ')
    if words[1] == '.I' then
        idx = idx +1 
        train_set = {}
        print(idx)
    else
        if words[1] ~= '.W' then
            for _,word in pairs(words) do
                if filt_map[Word2ID[word]] then
                    train_set[ Word2ID[word] ] = (train_set[ Word2ID[word] ] or 0 )+ 1
                end
            end
        end
    end
    if #words == 0 then
        if findElement(isTest,idx) then
            label = tostring(train_label_set[idx])
            line = ''
            --print(test_label_set[idx])
            for k,v in pairs(train_set) do
                if filt_map[tonumber(k)] then
                    line = line ..' '..filt_map[tonumber(k)]..':'..v
                end
            end
            if string.len(line) > 0 then
                fout_test:write(label..line..'\n')
            end
        else
            label = tostring(train_label_set[idx])
            line = ''
            --print(test_label_set[idx])
            for k,v in pairs(train_set) do
                if filt_map[tonumber(k)] then
                    line = line ..' '..filt_map[tonumber(k)]..':'..v
                end
            end
            if string.len(line) > 0 then
                fout_train:write(label..line..'\n')
            end   
        end 
    end
end
fin:close()
fout_train:close()
fout_test:close()
