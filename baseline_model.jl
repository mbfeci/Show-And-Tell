for p in ("Knet","Images")
    Pkg.installed(p) == nothing && Pkg.add(p)
end

using Knet, Images

function main(args="")
	
	data = Any[]

	data8kTokens = readdlm(string(pwd(), "/Flickr8kText/Flickr8k.token.txt"), String;comments=false);
	data8kWords = data8kTokens[:,2:end];
	push!(data, data8kWords)

	data30kTokens = readdlm(pwd()*"/Flickr30kText/results_20130124.token", String;comments=false);
	data30kWords = data30kTokens[:,2:end];
	push!(data, data30kWords)
	
	vocab = createVocabulary(data)
	
	
	w = initweights(KnetArray{Float32}, 1000, 9408, length(vocab) , 0.1);

    state = initstate(KnetArray{Float32},1000,1)
	
	#Directory of Flickr30k Dataset
	dirImage = pwd()*"/flickr30k-images"
	
	#=Directory of the caption of that image
	
	=#
	images = readdir(dirImage)
	
	#=Training in here
	for s in images
		img = processImage(dirImage*"/"*s)
		
	
	end
	=#
	
	#Directory of a random image from Flickr30:
	dirOneImage = dirImage*"/1000092795.jpg"
	
	#Caption of that particular image:
	generateCaption(dirOneImage, w, state, vocab, 100)
	
end


function generateCaption(dirImage, w, state, vocab, nwords)
	index_to_word = Array(String, length(vocab));
    for (k,v) in vocab; index_to_word[v] = k; end
	
	img = processImage(KnetArray{Float32}, dirImage)
	inputRNN = mat(simpleConvNet(w,img))'
	input = predictNextWord(w, state, inputRNN)
	input = getStartWord(KnetArray{Float32}, vocab)
	index = 1
	for t in 1:nwords
		ypred = predictNextWord(w, state, input*w[13])
		input[index] = 0

        index = sample(exp(logp(ypred)))
		
		if(index_to_word[index]=="</s>")
			println()
			return;
		end
		
		print(index_to_word[index], " ")
		input[index] = 1
    end
end


function sample(p)
    p = convert(Array,p)
    r = rand()
    for c = 1:length(p)
        r -= p[c]
        r < 0 && return c
    end
end


function processImage(atype, dir)

	img = load(dir)
    img = Images.imresize(img, (224,224))
	img = permutedims(channelview(img), (3,2,1))
	img = convert(Array{Float32}, img)
	img = reshape(img[:,:,1:3],(224,224,3,1))
	
	img = convert(atype, img)
	
end


#=
function loss(w,s,sequence,range=1:length(sequence)-1)
    total = 0.0; count = 0
    input = sequence[first(range)]
    for t in range
		prediction = predict(w,s,sequence[t])
		total += sum(sequence[t+1].*logp(prediction, 2))
		count += size(sequence[t],1)
    end
    return -total / count
end
=#


function getStartWord(atype, vocab)
	x = zeros(Float32, 1,length(vocab))
	x[vocab["<s>"]] = 1
	x = convert(atype, x)
	return x
end


function getEndWord(atype, vocab)
	x = zeros(Float32, 1,length(vocab))
	x[vocab["</s>"]] = 1
	x = convert(atype, x)
	return x
end


function initstate(atype, hidden, batchsize)
    state = Array(Any,1)
	state[1] = randn(batchsize,hidden)
    return map(s->convert(atype,s), state)
end


function oneHotVector(atype, s, vocab)
	x = zeros(Float32, 1,length(vocab))
	
	index = get(vocab,s,-1)
	
	if index != -1
		x[index] = 1
	end
	
	x = convert(atype,x)
	return x
end


function initweights(atype, hidden, input, vocab, winit)
    #=VGG ConvNet parameters
	w = Any[ -0.1+0.2*rand(Float32, 3,3,3,64),  zeros(Float32,1,1,64,1),
	-0.1 + 0.2*rand(Float32,3,3,64,128), zeros(Float32,1,1,128,1),
	-0.1 + 0.2*rand(Float32,3,3,128,256), zeros(Float32,1,1,256,1),		
	-0.1 + 0.2*rand(Float32,3,3,256,256), zeros(Float32,1,1,256,1),
	-0.1 + 0.2*rand(Float32,3,3,256,512), zeros(Float32,1,1,512,1),
	-0.1 + 0.2*rand(Float32,3,3,512,512), zeros(Float32,1,1,512,1)]
	=#

	
	#For baseline model, a simple ConvNet is used.
	w = Any[ -0.1+0.2*rand(Float32, 3,3,3,6),  zeros(Float32,1,1,6,1),
	-0.1 + 0.2*rand(Float32,3,3,6,12), zeros(Float32,1,1,12,1),
	-0.1 + 0.2*rand(Float32,3,3,12,12), zeros(Float32,1,1,12,1)]
	
	Whh = eye(hidden);
	Bhh = zeros(1,hidden);
	Wxh = winit*randn(input,hidden);
	Bxh = zeros(1,hidden);
	Wo = winit*randn(hidden,vocab);
	bo = zeros(1,vocab);
	We = winit*rand(vocab, input)
	append!(w, [Whh, Bhh, Wxh, Bxh, Wo, bo, We])

	w = map(k->convert(atype,k), w)
    return w
end


#Input size is 224x224x3x1
#Output size will be 28x28x12x1
function simpleConvNet(w, x)
	x1 = pool(relu(conv4(w[1], x; stride = 1, padding = 1).+w[2]);window = 2)
	x2 = pool(relu(conv4(w[3], x1; stride = 1, padding = 1).+w[4]);window = 2)
	x3 = pool(relu(conv4(w[5], x2; stride = 1, padding = 1).+w[6]);window = 2)
	
	return x3
end


function rnn(w,s,x)
	s[1] = tanh(s[1]*w[7] .+ w[8] + x*w[9] .+ w[10]);
	return s
end


function predictNextWord(w, s, x)
	s = rnn(w,s,x)
	return s[1]*w[11] .+ w[12];
end


function createVocabulary(dataArray)
    vocab = Dict{String,Int}()
	get!(vocab, "<s>", 1)       #Special start token
	get!(vocab, "</s>", 2)      #Special end token
	for data in dataArray
		for word in data
			get!(vocab, word, length(vocab)+1)
		end
	end
    return vocab
end

main()