for p in ("Knet","Images","ArgParse")
    Pkg.installed(p) == nothing && Pkg.add(p)
end

using Knet, Images, ArgParse

include("parser.jl");

const imgurl = "https://github.com/BVLC/caffe/raw/master/examples/images/cat.jpg"
const caption_files = ["data/Flickr30kText/results_20130124.token"];

function main(args=ARGS)
	
	s = ArgParseSettings()
    s.description="vgg.jl (c) Mehmed Burak Demirci, 2017. Implementation of the model in the following paper: https://arxiv.org/pdf/1411.4555.pdf ."
    # s.exc_handler=ArgParse.debug_handler
    @add_arg_table s begin
		("image"; default=imgurl; help="Image file or URL.")
		("--datafiles"; nargs='+'; help="If provided, use first file for training, second for dev, others for test.")
        ("--generate"; arg_type=Int; default=100; help="If non-zero generate given number of characters.")
		("--hidden"; nargs='*'; arg_type=Int; help="sizes of hidden layers of multilayer LSTM, e.g. --hidden 512 256 for a net with two LSTM layers.")
        ("--epochs"; arg_type=Int; default=3; help="Number of epochs for training.")
        ("--batchsize"; arg_type=Int; default=1; help="Number of sequences to train on in parallel.")
        ("--seqlength"; arg_type=Int; default=25; help="Number of steps to unroll the network for.")
		("--embed"; arg_type=Int; default=1000; help="Size of the embedded word vector.")
        ("--decay"; arg_type=Float64; default=0.5; help="Learning rate decay.")
        ("--lr"; arg_type=Float64; default=1e-1; help="Initial learning rate.")
        ("--gclip"; arg_type=Float64; default=3.0; help="Value to clip the gradient norm at.")
        ("--winit"; arg_type=Float64; default=0.1; help="Initial weights set to winit*randn().")
        ("--gcheck"; arg_type=Int; default=0; help="Check N random gradients.")
        ("--atype"; default=(gpu()>=0 ? "KnetArray{Float32}" : "Array{Float32}"); help="array type: Array for cpu, KnetArray for gpu")
    end
	
	o = parse_args(s; as_symbols=true)
	
	#=
	o = Dict();
	o[:atype] = KnetArray{Float32};
	o[:hidden] = [512];
	o[:embed] = [512];
	o[:batchsize] = 1;
	o[:lr] = 0.1;
	o[:decay] = 0.5;
	o[:winit] = 0.1;
	o[:epochs] = 3;
	o[:generate] = 100;
	o[:image] = imgurl
	=#
	dicts, word2index = parse_files(caption_files);
	println("Dictionary and vocabulary are created, the size of vocabulary is ", length(word2index));
	w = initweights(o[:atype], o[:hidden], o[:embed], length(word2index) , o[:winit]);

    state = initstate(o[:atype],o[:hidden],o[:batchsize])
	
	ids, data = prepare_data(dicts[1], word2index);

	train(w, ids, data, word2index, o)
	
	#Caption of that particular image:
	if o[:generate] > 0
		img = processImage(o[:image], zeros(Float32,224,224,3,1))
		new_state = initstate(o[:atype],o[:hidden],o[:batchsize])
		generateCaption(img, w, new_state, word2index, o[:generate])
	end
end

#Assuming seq is: [word1 word2 word3 ...]
function loss(w,s,img,sequence,range=1:length(sequence)-1)
    total = 0.0; count = 0
	
	#Do we want to train CNN as well?
	encoded_image = simpleConvNet(img)
	rnn(w,s,encoded_image);
	
    for t in range
		prediction = rnn(w,s,sequence[t])
		total += sum(sequence[t+1].*logp(prediction, 2))
		count += size(sequence[t],1)
    end
    return -total / count
end


lossgradient = grad(loss);

function train(w, ids, sequence, word2index, o)
	state = initstate(o[:atype],o[:hidden],o[:batchsize])
	lr = o[:lr]
	prev_id = 0
	for epoch = 1:o[:epochs]
		index = 1
		for id in ids
			if prev_id!=id
				imageloc = "data/flickr30k-images/$id.jpg";
				img = processImage(imageloc, zeros(Float32,224,224,3,1))
			end
			
			for seq in sequence[index]
				
				sentence = convert(o[:atype], seq)
				
				gloss = lossgradient(w,state,img,sentence)
		
				update!(w, gloss; lr=lr, gclip=o[:gclip])
				
			end
			println("$id is finished.");
			prev_id = id
		end
		if epochs % 5 == 0
			lr *= o[:decay]
		end
	end
	
end


function prepare_data(dict, word2index)
	vocab_size = length(word2index)
	ids = Array(Int64, 5*length(dict));
	data = Array{Any,1}()
	index = 1
	for seq in dict
		id = seq[1];
		no = seq[2][1];
		words = seq[2][2];
		
		sentence = [ falses(1, vocab_size) for i=1:length(words) ] # using BitArrays
		
		for k = 1:length(words)
			sentence[k][word2index[words[k]]] = 1
		end
		
		push!(data, sentence);
		ids[index] = id;
		index += 1;
	end
	
	return ids, data
end



function generateCaption(dirImage, w, state, vocab, nwords)
	index2word = Array(String, length(vocab));
    	for (k,v) in vocab; index2word[v] = k; end
	
	img = processImage(dirImage, zeros(Float32,224,224,3,1))
	inputRNN = simpleConvNet(w,img)
	
	input = rnn(w, state, img)
	input = getStartWord(KnetArray{Float32}, vocab)
	index = 1
	for t in 1:nwords
		ypred = rnn(w, state, input*w[13])
		input[index] = 0
        	index = sample(exp(logp(ypred)))
		if(index2word[index]=="</s>")
			println()
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


function initweights(atype, hidden, input, vocab, winit)
	w = init_cnn_weights(winit, input);
	append!(w, init_rnn_weights(hidden, vocab, input))
    return map(k->convert(atype,k), w)
end


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

#Input size is 224x224x3x1
function simpleConvNet(w, x)
	x1 = pool(relu(conv4(w[1], x; stride = 1, padding = 2).+w[2]);window = 2, stride = 2)
	x2 = pool(relu(conv4(w[3], x1; stride = 1, padding = 2).+w[4]);window = 2, stride = 2)
	x3 = pool(relu(conv4(w[5], x2; stride = 1, padding = 2).+w[6]);window = 2, stride = 2)
	x4 = pool(relu(conv4(w[7], x3; stride = 1, padding = 2).+w[8]);window = 2, stride = 2)
	
	
	return mat(x4)'*w[9] + w[10];
end


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


function rnn(w,s,x; start = 10)
	input = x*w[end]
	for i=1:2:length(s)
		(s[i],s[i+1]) = lstm(w[start + i],w[start + i+1],s[i],s[i+1],input)
		input = s[i]
	end
	return s[end-2]*w[s] .+ w[end-1]
end


main()
