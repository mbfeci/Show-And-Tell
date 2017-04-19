
for p in ("Knet","Images","ArgParse","ImageMagick","MAT")#, "JLD")
    Pkg.installed(p) == nothing && Pkg.add(p)
end

using Knet, Images, ArgParse, MAT#, JLD

include("parser.jl");

const imgurl = "https://github.com/BVLC/caffe/raw/master/examples/images/cat.jpg"
const vggurl = "http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-16.mat"
const caption_file = "data/Flickr30k/Flickr30kText/results_20130124.token";
const LAYER_TYPES = ["conv", "relu", "pool", "fc", "prob"]

function main(args=ARGS)
	
	s = ArgParseSettings()
    s.description="vgg.jl (c) Mehmed Burak Demirci, 2017. Implementation of the model in the following paper: https://arxiv.org/pdf/1411.4555.pdf ."
    # s.exc_handler=ArgParse.debug_handler
    @add_arg_table s begin
		("image"; default=imgurl; help="Image file or URL.")
		("--datafiles"; nargs='+'; help="If provided, use first file for training, second for dev, others for test.")
		("--model"; default=Knet.dir("data","imagenet-vgg-verydeep-16.mat"); help="Location of the model file")
        ("--generate"; arg_type=Int; default=100; help="If non-zero generate given number of characters.")
		("--hidden"; nargs='*'; arg_type=Int; default=[512]; help="sizes of hidden layers of multilayer LSTM, e.g. --hidden 512 256 for a net with two LSTM layers.")
        ("--epochs"; arg_type=Int; default=1; help="Number of epochs for training.")
        ("--batchsize"; arg_type=Int; default=20; help="Number of sequences to train on in parallel.")
        ("--seqlength"; arg_type=Int; default=25; help="Number of steps to unroll the network for.")
		("--embed"; arg_type=Int; default=512; help="Size of the embedded word vector.")
        ("--decay"; arg_type=Float64; default=0.5; help="Learning rate decay.")
        ("--lr"; arg_type=Float64; default=1e-1; help="Initial learning rate.")
        ("--gclip"; arg_type=Float64; default=3.0; help="Value to clip the gradient norm at.")
        ("--winit"; arg_type=Float64; default=0.1; help="Initial weights set to winit*randn().")
        ("--gcheck"; arg_type=Int; default=0; help="Check N random gradients.")
		("--fast"; action=:store_true; help="skip loss printing for faster run")
        ("--atype"; default=(gpu()>=0 ? KnetArray{Float32} : Array{Float32}); help="array type: Array for cpu, KnetArray for gpu")
    end
	
	o = parse_args(s; as_symbols=true)
	
	#=
	o = Dict();
	o[:model] = Knet.dir("data","imagenet-vgg-verydeep-16.mat")
	o[:atype] = KnetArray{Float32};
	o[:hidden] = [512];
	o[:embed] = 512;
	o[:batchsize] = 20;
	o[:lr] = 0.1;
	o[:decay] = 0.5;
	o[:winit] = 0.1;
	o[:epochs] = 3;
	o[:generate] = 100;
	o[:image] = imgurl
	o[:gclip] = 3.0
	o[:fast] = false
	=#
	
	println("opts=",[(k,v) for (k,v) in o]...)
	#=
	if !isfile(o[:model])
        println("Should I download the VGG model (492MB)? Enter 'y' to download, anything else to quit.")
        readline() == "y\n" || return
        download(vggurl,o[:model])
    end
	
	info("Reading $(o[:model])")
    vgg = matread(o[:model])
    vgg_params = get_params(vgg)
    global convnet = get_convnet(vgg_params...)
	global averageImage = convert(Array{Float32},vgg["meta"]["normalization"]["averageImage"])
	=#

	dict, word2index = parse_file(caption_file);
	info("Dictionary and vocabulary are created");
	info("The size of vocabulary is $(length(word2index))");
	w = initweights(o[:atype], o[:hidden], o[:embed], length(word2index) , o[:winit]);

    state = initstate(o[:atype],o[:hidden],o[:batchsize])
	
	ids, data = minibatch(dict, word2index, o[:batchsize]);

	dicts = 0
	Knet.knetgc(); gc();
	#Execute only once:
	save_vgg_outputs(convnet, ids, data, o[:batchsize])
	
	train(w, state, ids, data, word2index, o)

	#Caption of that particular image:
	if o[:generate] > 0
		new_state = initstate(o[:atype],o[:hidden],1)
		generateCaption(o[:image], w, new_state, word2index, o[:generate])
	end

end

#Assuming seq is: [word1 word2 word3 ...]
function loss(w,s,cout,sequence,range=1:length(sequence)-1)
    total = 0.0; count = 0
	
	rnn(w,s,cout*w[1]);
	
	prediction = rnn(w,s,getStartWord(size(sequence[1],1),size(sequence[1],2))*w[end])
	total += sum(sequence[1].*logp(prediction, 2))
	
    for t in range
		prediction = rnn(w,s,sequence[t]*w[end])
		total += sum(sequence[t+1].*logp(prediction, 2))
		count += size(sequence[t],1)
    end
	
	prediction = rnn(w,s,getEndWord(size(sequence[1],1),size(sequence[1],2))*w[end])
	total += sum(sequence[end].*logp(prediction, 2))
	count += size(sequence[end],1)
	
    return -total / count
end

lossgradient = grad(loss);

function train(w, state, ids, sequence, word2index, o; cnnout = 4096)
	lr = o[:lr]
	prev_id = 0
	
	for epoch = 1:o[:epochs]
		for index in 1:length(sequence)
			cout = Array(Float32, o[:batchsize], cnnout)
			for batchno in 1:o[:batchsize]
				id = ids[batchno,index]
				feature_directory = "./data/Flickr30k/VGG/features/$(id).jld"
				if isfile(feature_directory)
					cout[batchno, :] = load(feature_directory, "feature")
				else
					imageloc = "data/Flickr30k/flickr30k-images/$id.jpg";
					cout[batchno, :] = convert(Array{Float32}, convnet(processImage(imageloc, averageImage)))
					save(feature_directory, "feature", cout[batchno,:])
				end
			end
			
			cout = convert(o[:atype], cout)
			sentence = map(k->convert(o[:atype], k),sequence[index])
			
			gloss = lossgradient(w,copy(state),cout,sentence)
			
			for j in 1:length(w)
				update!(w[j], gloss[j]; lr=lr)
			end
			
			if index%100==0
				@printf("%d is trained %0.3f%% of epoch is completed.\n",index, index/length(sequence)*100)
				println("$(Knet.gpufree()) GPU memory left");
				println("loss in this sentence is: ", loss(w,copy(state),cout,sentence))
			end
			
		end
		
		if epochs % 5 == 0
			lr *= o[:decay]
		end
		
	end
end

#=
function prepare_data(dict, word2index)
	vocab_size = length(word2index)
	ids = Array(Int64, 5*length(dict));
	data = Array{Any,1}()
	index = 1
	for seq in dict
		id = seq[1];
		no = seq[2][1];
		words = seq[2][2];
		
		sentence = [ falses(1, vocab_size) for i=1:length(words)+2 ] # using BitArrays
		
		sentence[1][word2index["<s>"]] = 1
		for k = 1:length(words)
			sentence[k][word2index[words[k]]] = 1
		end
		sentence[end][word2index["</s>"]] = 1
		
		push!(data, sentence);
		ids[index] = id;
		index += 1;
	end
	
	return ids, data
end
=#

function save_vgg_outputs(convnet,ids,data,batchsize)
	
	for index in 1:length(data)
		for batchno in 1:batchsize
			id = ids[batchno,index]
			filename = "./data/Flickr30k/VGG/features/$id.jld";
			
			if !isfile(filename)
				imageloc = "data/Flickr30k/flickr30k-images/$id.jpg";
				img = processImage(imageloc, averageImage)
				cout = convert(Array{Float32},convnet(img))
				save(filename,"feature",cout)
			end
		end
		
		println("The batch with no ", index, " is successfully saved");
	end
	println("DONE!");
end


function minibatch(dict, word2index, batchsize)
	vocab_size = length(word2index)
	
	ids = Array(Int64, batchsize, div(length(dict),batchsize)); #An upperbound for the size of the ids
	data = Array{Any,1}()
	index = 1
	count = 1
	while true
		index>length(dict)-batchsize+1 && break
		
		seq = dict[index];
		id = seq[1];
		len = seq[2];
		words = seq[3];
		if dict[index+batchsize-1][2] != len
			index += 1;
			continue;
		end
		
		sentence = [ falses(batchsize, vocab_size) for j=1:length(words) ] # using BitArrays
		for i in index:index+batchsize-1
			seq = dict[i];
			id = seq[1];
			len = seq[2];
			words = seq[3];
			for k = 1:len
				sentence[k][i-index+1,word2index[words[k]]] = 1
			end
			
			ids[i-index+1,count] = id;
		end
		push!(data, sentence);

		count += 1
		index += batchsize
	end
	
	info("Minibatch completed with $count batches of size $batchsize")
	return ids, data
end

function generateCaption(dirImage, w, state, vocab, nwords)
	index2word = Array(String, length(vocab));
    	for (k,v) in vocab; index2word[v] = k; end
	
	img = processImage(dirImage, averageImage)
	inputRNN = convnet(img)*w[1]
	
	input = rnn(w, state, inputRNN)
	input = getStartWord(KnetArray{Float32}, vocab)
	index = 1
	for t in 1:nwords
		ypred = rnn(w, state, input*w[end])
		input[index] = 0
        	index = sample(exp(logp(ypred)))
		if(index2word[index]=="</s>")
			println(".")
			return;
		end
		print(index2word[index], " ")
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


function processImage(img, averageImage)
    if contains(img,"://")
        info("Downloading $img")
        img = download(img)
    end
    a0 = load(img)
    new_size = ntuple(i->div(size(a0,i)*224,minimum(size(a0))),2)
    a1 = Images.imresize(a0, new_size)
    i1 = div(size(a1,1)-224,2)
    j1 = div(size(a1,2)-224,2)
    b1 = a1[i1+1:i1+224,j1+1:j1+224]
    c1 = permutedims(channelview(b1), (3,2,1))
    d1 = convert(Array{Float32}, c1)
    e1 = reshape(d1[:,:,1:3], (224,224,3,1))
    f1 = (255 * e1 .- averageImage)
    g1 = permutedims(f1, [2,1,3,4])
    x1 = KnetArray(g1)
end


function getStartWord(batchsize, vocabsize; atype = KnetArray{Float32})
	x = zeros(Float32, batchsize,vocabsize)
	x[:,1] = 1
	x = convert(atype, x)
	return x
end


function getEndWord(batchsize, vocabsize; atype = KnetArray{Float32})
	x = zeros(Float32, batchsize,vocabsize)
	x[:,2] = 1
	x = convert(atype, x)
	return x
end


# state[2k-1]: hidden for the k'th lstm layer
# state[2k]: cell for the k'th lstm layer
function initstate(atype, hidden_layers, batchsize)
	nlayers = length(hidden_layers);
	state = Array(Any, 2*nlayers);
    for k = 1:nlayers
        state[2k-1] = zeros(Float32, batchsize, hidden_layers[k]);
		state[2k] = zeros(Float32, batchsize, hidden_layers[k]);
    end
    return map(k->convert(atype,k), state)
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


function initweights(atype, hidden, input, vocab, winit, cout=4096)
	#w = init_cnn_weights(winit, input);
	w = [xavier(cout,input)];
	append!(w,init_rnn_weights(hidden, vocab, input))
	#append!(w, init_rnn_weights(hidden, vocab, input))
    return map(k->convert(atype,k), w)
end


#=
function xavier(a...)
    w = rand(a...)
     # The old implementation was not right for fully connected layers:
     # (fanin = length(y) / (size(y)[end]); scale = sqrt(3 / fanin); axpb!(rand!(y); a=2*scale, b=-scale)) :
    if ndims(w) < 2
        error("ndims=$(ndims(w)) in xavier")
    elseif ndims(w) == 2
        fanout = size(w,1)
        fanin = size(w,2)
    else
        fanout = size(w, ndims(w)) # Caffe disagrees: http://caffe.berkeleyvision.org/doxygen/classcaffe_1_1XavierFiller.html#details
        fanin = div(length(w), fanout)
    end
    # See: http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf
    s = sqrt(2 / (fanin + fanout))
    w = 2s*w-s
end
=#

function init_rnn_weights(hidden, vocab, embed)
    model = Array(Any, 2*length(hidden)+3)
    X = embed
    for k = 1:length(hidden)
        H = hidden[k]
        model[2k-1] = xavier(X+H, 4H)
        model[2k] = zeros(1, 4H)
        model[2k][1:H] = 1 # forget gate bias = 1
        X = H
    end

    model[end-2] = xavier(hidden[end],vocab)
    model[end-1] = zeros(1,vocab)			
	model[end] = xavier(vocab,embed)	#We (word embedding vector)
    return model
end

#=
function init_cnn_weights(winit, embed)

	#=VGG ConvNet parameters
	w = Any[ -0.1+winit*rand(Float32, 3,3,3,64),  zeros(Float32,1,1,64,1),
	-0.1 + winit*rand(Float32,3,3,64,128), zeros(Float32,1,1,128,1),
	-0.1 + winit*rand(Float32,3,3,128,256), zeros(Float32,1,1,256,1),		
	-0.1 + winit*rand(Float32,3,3,256,256), zeros(Float32,1,1,256,1),
	-0.1 + winit*rand(Float32,3,3,256,512), zeros(Float32,1,1,512,1),
	-0.1 + winit*rand(Float32,3,3,512,512), zeros(Float32,1,1,512,1)]
	=#
	#For baseline model, a simple ConvNet is used.
	w = Any[ -0.1+winit*rand(Float32, 5,5,3,10),  zeros(Float32,1,1,10,1),
	-0.1 + winit*rand(Float32,5,5,10,10), zeros(Float32,1,1,10,1),
	-0.1 + winit*rand(Float32,5,5,10,10), zeros(Float32,1,1,10,1),
	-0.1 + winit*rand(Float32,5,5,10,10), zeros(Float32,1,1,10,1),
	-0.1 + winit*rand(Float32,1960,embed), zeros(Float32,1,embed)]; 

	return w
end
=#

#=
#Input size is 224x224x3x1
function simpleConvNet(w, x)
	x1 = pool(relu(conv4(w[1], x; stride = 1, padding = 2).+w[2]);window = 2, stride = 2)
	x2 = pool(relu(conv4(w[3], x1; stride = 1, padding = 2).+w[4]);window = 2, stride = 2)
	x3 = pool(relu(conv4(w[5], x2; stride = 1, padding = 2).+w[6]);window = 2, stride = 2)
	x4 = pool(relu(conv4(w[7], x3; stride = 1, padding = 2).+w[8]);window = 2, stride = 2)

	return mat(x4)'*w[9] .+ w[10];
end
=#

function lstm(weight,bias,hidden,cell,input)
    gates   = hcat(input,hidden) * weight .+ bias
    hsize   = size(hidden,2)
    forget  = sigm(gates[:,1:hsize])
    ingate  = sigm(gates[:,1+hsize:2hsize])
    outgate = sigm(gates[:,1+2hsize:3hsize])
    change  = tanh(gates[:,1+3hsize:end])
    cell    = cell .* forget + ingate .* change
    hidden  = outgate .* tanh(cell)
    return (hidden,cell)
end


function rnn(w,s,input; start = 1)
	for i=1:2:length(s)
		(s[i],s[i+1]) = lstm(w[start + i],w[start + i+1],s[i],s[i+1],input)
		input = s[i]
	end
	return input*w[end-2] .+ w[end-1]
end


# This procedure makes pretrained MatConvNet VGG parameters convenient for Knet
# Also, if you want to extract features, specify the last layer you want to use
function get_params(CNN; last_layer="fc7")
    layers = CNN["layers"]
    weights, operations, derivatives = [], [], []

    for l in layers
        get_layer_type(x) = startswith(l["name"], x)
        operation = filter(x -> get_layer_type(x), LAYER_TYPES)[1]
        push!(operations, operation)
        push!(derivatives, haskey(l, "weights") && length(l["weights"]) != 0)

        if derivatives[end]
            w = l["weights"]
            if operation == "conv"
                w[2] = reshape(w[2], (1,1,length(w[2]),1))
            elseif operation == "fc"
                w[1] = transpose(mat(w[1]))
            end
            push!(weights, w)
        end

        last_layer != nothing && get_layer_type(last_layer) && break
    end

    map(w -> map(KnetArray, w), weights), operations, derivatives
end

# get convolutional network by interpreting parameters
function get_convnet(weights, operations, derivatives)
    function convnet(xs)
        i, j = 1, 1
        num_weights, num_operations = length(weights), length(operations)
        while i <= num_operations && j <= num_weights
            if derivatives[i]
                xs = forw(xs, operations[i], weights[j])
                j += 1
            else
                xs = forw(xs, operations[i])
            end

            i += 1
        end
        return xs'
    end
end

# convolutional network operations
convx(x,w) = conv4(w[1], x; padding=1, mode=1) .+ w[2]
relux = relu
poolx = pool
probx(x) = x
fcx(x,w) = w[1] * mat(x) .+ w[2]
tofunc(op) = eval(parse(string(op, "x")))
forw(x,op) = tofunc(op)(x)
forw(x,op,w) = tofunc(op)(x,w)


main()
