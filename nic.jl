
#=
Beam search
Model is training very slowly, can not even properly learn the training set
Loss is stuck at 1.88's
Should I increase the hidden size? Or should I train more? Or is something wrong?
=#

for p in ("Knet","Images","ArgParse","ImageMagick","MAT", "JSON", "JLD")
    Pkg.installed(p) == nothing && Pkg.add(p)
end

using Knet, Images, ArgParse, MAT, JLD, JSON

# To be able to load/save KnetArrays:
if Pkg.installed("JLD") != nothing
    import JLD: writeas, readas
    type KnetJLD; a::Array; end
    writeas(c::KnetArray) = KnetJLD(Array(c))
    readas(d::KnetJLD) = KnetArray(d.a)
end

include("parser.jl");

const imgurl = "https://github.com/BVLC/caffe/raw/master/examples/images/cat.jpg"
const dogurl = "http://cdn.wallpapersafari.com/18/70/GHrovc.jpg"
const vggurl = "http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-16.mat"
const caption_file = "data/Flickr30k/Flickr30kText/results_20130124.token"
const caption_file_cocotrain =  "data/MSCOCO/annotations/captions_train2014.json"
const caption_file_cocoval = "data/MSCOCO/annotations/captions_val2014.json"
const LAYER_TYPES = ["conv", "relu", "pool", "fc", "prob"]


function main(args=ARGS)
	
	s = ArgParseSettings()
    s.description="vgg.jl (c) Mehmed Burak Demirci, 2017. Implementation of the model in the following paper: https://arxiv.org/pdf/1411.4555.pdf ."
    # s.exc_handler=ArgParse.debug_handler
    @add_arg_table s begin
		("image"; default=imgurl; help="Image file or URL.")
		("--cnnout"; default = 4096; help="Length of the image feature vectors")
		("--imgid"; arg_type=Int; default=6734417; help="id of the image to be captioned")
		("--notrain"; action=:store_true; help="skip training")
		("--transfer"; action=:store_true; help="transfer learning.")
		("--bestmodel"; default = "./model/lol.jld"; help="The location of the parameters of the best model")
		("--datafiles"; nargs='+'; help="If provided, use first file for training, second for dev, others for test.")
		("--model"; default=Knet.dir("data","imagenet-vgg-verydeep-16.mat"); help="Location of the model file")
        ("--generate"; arg_type=Int; default=100; help="If non-zero generate given number of characters.")
		("--hidden"; nargs='*'; arg_type=Int; default=[512]; help="sizes of hidden layers of multilayer LSTM, e.g. --hidden 512 256 for a net with two LSTM layers.")
        ("--epochs"; arg_type=Int; default=10; help="Number of epochs for training.")
        ("--batchsize"; arg_type=Int; default=64; help="Number of sequences to train on in parallel.")
        ("--seqlength"; arg_type=Int; default=25; help="Number of steps to unroll the network for.")
		("--embed"; arg_type=Int; default=512; help="Size of the embedded word vector.")
        ("--decay"; arg_type=Float64; default=0.5; help="Learning rate decay.")
        ("--lr"; arg_type=Float64; default=1e-1; help="Initial learning rate.")
        ("--gclip"; arg_type=Float64; default=3.0; help="Value to clip the gradient norm at.")
        ("--winit"; arg_type=Float64; default=0.1; help="Initial weights set to winit*randn().")
        ("--gcheck"; arg_type=Int; default=0; help="Check N random gradients.")
		("--fast"; action=:store_true; help="skip loss printing for faster run")
        ("--atype"; default=(gpu()>=0 ? KnetArray{Float32} : Array{Float32}); help="array type: Array for cpu, KnetArray for gpu")
		("--savefile"; default="./model/dropout.jld" ;help="Save final model to file")
		("--check"; action=:store_true; help="Check if the ids are correctly matched to the captions")
		("--normalize"; action=:store_true; help="Use normalized VGG features.")
		("--dropout"; arg_type=Float64; default=0.0; help="Dropout probability.")
		("--bleu"; action=:store_true; help="Create files for BLEU scores.")
	end
	
	o = parse_args(s; as_symbols=true)
	
	#=
	o = Dict();
	o[:model] = Knet.dir("data","imagenet-vgg-verydeep-16.mat")
	o[:atype] = KnetArray{Float32};
	o[:hidden] = [512];
	o[:embed] = 512;
	o[:batchsize] = 50;
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
	
	dict, vocab = parse_file(caption_file);
	global word2index = vocab
	
	train_dict, valid_dict, test_dict = splitdata(dict, 1000, 1000)
	
	global	index2word = Array(String, length(word2index));
    for (k,v) in word2index; index2word[v] = k; end
	

	info("Dictionary and vocabulary are created");
	info("The size of vocabulary is $(length(word2index))");
	
	if !o[:transfer]
		w = initweights(o[:atype], o[:hidden], o[:embed], length(word2index) , o[:winit]);
	else
		info("Loading best model...")
		w = load(o[:savefile], "model")
	end
	
	#w = load("./model/nic.jld", "model 1")
    state = initstate(o[:atype],o[:hidden],o[:batchsize])
	
	valstate = initstate(o[:atype], o[:hidden], 1)
	
	train_ids, train_data = minibatch(train_dict, o[:batchsize]);
	
	valid_ids, valid_data = minibatch(valid_dict, 1);
	
	test_ids, test_data = minibatch(test_dict, 1);
	
	info("Loading features into memory...")
	
	
	function load_normalized_features()
		return load("./data/Flickr30k/VGG/features/normalized_features.jld", "normalized_feature_dict")
	end
	
	function load_features()
		return load("./data/Flickr30k/VGG/features/features.jld", "feature_dict")
	end
	
	global features = o[:normalize] ? load_normalized_features() : load_features()	
	
	dict = 0; train_dict = 0; valid_dict = 0; test_dict = 0;
	Knet.knetgc(); gc();
	
	#val_loss = calculate_loss(w, copy(valstate), o[:cnnout], valid_data, valid_ids, 1, o[:check])		
	Knet.knetgc(); gc();
	
	#println("Validation loss: ", val_loss)
	
	o[:bleu] && bleu(test_data, test_ids, w, copy(valstate), beamsize=20)
	
	#Execute only once:
	#save_Flickr30k_features()
	#save_coco_train_features()
	#save_coco_val_features()
	
	opts = init_params(w)
	
	println("Training is starting...")
	
	!o[:notrain] && train(w, state, valstate, train_ids, train_data, valid_ids, valid_data, word2index, o, opts)

	#Caption of that particular image:
	if o[:generate] > 0
		new_state = initstate(o[:atype],o[:hidden],1)
		generate_caption(o[:image], w, copy(new_state), word2index, o[:generate]; normalized = o[:normalize])
		println("Generating caption of the image with id: $(o[:imgid])")
		generate_caption(o[:imgid], w, copy(new_state), word2index,	o[:generate])
	end

end

function bleu(test_data,test_ids,w,state;batchsize=1, beamsize=5, cnnout=4096)
	refs = [open("ref$i","w") for i=1:5]
	id_array = Array(Int64,1000);
	id_dict = Dict{Int,Int}();
	count = 1
	
	for id in test_ids
		if !haskey(id_dict, id)
			id_array[count] = id;
			count += 1
			get!(id_dict, id, 0);
		end
	end
	
	info("ids_array created")
	hypothesis = open("hypothesis.txt","w")
	
	count = 1
	for id in id_array
		generate_with_beam_search(id, w, copy(state), word2index, 40, beamsize; file=hypothesis)
		if count%100==0
			println(count, ". sentence is generated")
		end
		count += 1
	end
	
	close(hypothesis)
	
	id_strings = Dict{Int64,Any}();
	
	for index in 1:length(test_data)
		id = test_ids[index]
		get!(id_strings, id, [])
		push!(id_strings[id], test_data[index])
	end
	
	info("Strings are put into the dict")
	
	for i=1:5
		for id in id_array
			for j=1:length(id_strings[id][i])
				write(refs[i], index2word[id_strings[id][i][j][1]], " ")
			end
			write(refs[i], "\n")
		end
		close(refs[i]);
	end
	info("Strings are written into the reference files.")

end


function save_Flickr30k_features()
	println("Loading into memory...")
	
	flickr30k_features = Dict{Int64, Array{Float32}}()
	
	#flickr30k_features = load("./data/Flickr30k/VGG/features/feature_dict.jld", "feature_dict")
	
	dir = readdir("data/Flickr30k/flickr30k-images")
	
	index = 1
	
	for imgloc in dir
		filename = "data/Flickr30k/flickr30k-images/"*imgloc

		id = parse(Int64,imgloc[rsearch(imgloc,'_')+1:rsearch(imgloc,'.')-1])
		if !haskey(coco_features, id)
			#println("ID is $id")

			img = processImage(filename, averageImage)
		    cout = convnet(img)
			flickr30k_features[id] = convert(Array{Float32}, cout)
			if index%1000==0
				println("Saving...")
				save("./data/Flickr30k/VGG/features/feature_dict.jld", "feature_dict", coco_features)
			end

		end
		index%100==0 && println(index, " images are successfully saved");
		index += 1
	end
	
	println("Saving...")
	save("./data/Flickr30k/VGG/features/feature_dict.jld", "feature_dict", flickr30k_features)
	
	println("DONE!");

end

#=
function save_vgg_outputs(convnet,ids,data,batchsize)

	for index in 1:length(data)
		for batchno in 1:batchsize
			id = ids[batchno,index]
			filename = "./data/Flickr30k/VGG/features/$id.jld";
			
			if !isfile(filename)
				imageloc = "data/Flickr30k/flickr30k-images/$id.jpg";
				img = processImage(imageloc, averageImage)
				cout = convnet(img)
				save(filename,"feature",cout)
			end
		end
		if(index%100==1)
			id = ids[1,index]
			println("ID: $id")
			sentence = data[index]
			for word in sentence
				print(index2word[word[1]], " ")
			end	
			println()
		end
		index%100==0 && println("The batch with no ", index, " is successfully saved");
	end
	println("DONE!");
end
=#

#=
#Keeping all features in one field turned out to be more efficient
function save_feature_dict(ids)
	global features = Dict{Int64, KnetArray{Float32}}()
	
	count=0
	for id in ids
		if(!haskey(features,id))
			get!(features, id, load("./data/Flickr30k/VGG/features/$id.jld", "feature"))
			count += 1
			if count%100==0
				println(count)
			end
		end
	end
	
	info("Features are loaded into memory.")
	
	save("./data/Flickr30k/VGG/features/feature_dict.jld", "feature_dict", features)
	info("Feature dictionary is saved into jld file")
end
=#


#=
function save_normalized_features()
	
	normalized_features = Dict{Int64,KnetArray{Float32}}()
	
	for key in keys(features)
		normalized_features[key] = features[key]./(sum(features[key]))
	end
	
	info("Saving normalized features...")
	save("./data/Flickr30k/VGG/features/normalized_features.jld", "normalized_feature_dict", normalized_features)
	println("FINISHED")
	
end
=#



function save_coco_train_features()
	println("Loading into memory...")
	
	coco_features = Dict{Int64, Array{Float32}}()
	
	#coco_features = load("./data/MSCOCO/VGG/features/mscoco_feature_dict.jld", "feature_dict")
	
	dir = readdir("data/MSCOCO/train2014")
	
	index = 1
	
	for imgloc in dir
		filename = "data/MSCOCO/train2014/"*imgloc

		id = parse(Int64,imgloc[rsearch(imgloc,'_')+1:rsearch(imgloc,'.')-1])
		if !haskey(coco_features, id)
			#println("ID is $id")

			img = processImage(filename, averageImage)
		    cout = convnet(img)
			coco_features[id] = convert(Array{Float32}, cout)
			if index%1000==0
				println("Saving...")
				save("./data/MSCOCO/VGG/features/mscoco_feature_dict.jld", "feature_dict", coco_features)
			end

		end
		index%100==0 && println(index, " images are successfully saved");
		index += 1
	end
	
	println("Saving...")
	save("./data/MSCOCO/VGG/features/mscoco_feature_dict.jld", "feature_dict", coco_features)
	
	println("DONE!");

end


function save_coco_val_features()
	println("Loading into memory...")
	coco_features = Dict{Int64, Array{Float32}}()
	#coco_features = load("./data/MSCOCO/VGG/features/mscoco_val_feature_dict.jld", "feature_dict")
	
	dir = readdir("data/MSCOCO/val2014")
	
	index = 1
	
	for imgloc in dir
		filename = "data/MSCOCO/val2014/"*imgloc

		id = parse(Int64,imgloc[rsearch(imgloc,'_')+1:rsearch(imgloc,'.')-1])
		if !haskey(coco_features, id)
			#println("ID is $id")

			img = processImage(filename, averageImage)
		    cout = convnet(img)
			coco_features[id] = convert(Array{Float32}, cout)
			if index%1000==0
				println("Saving...")
				save("./data/MSCOCO/VGG/features/mscoco_val_feature_dict.jld", "feature_dict", coco_features)
			end
		end
		index%100==0 && println(index, " images are successfully saved");
		index += 1
	end
	
	println("Saving...")
	save("./data/MSCOCO/VGG/features/mscoco_val_feature_dict.jld", "feature_dict", coco_features)
	
	println("DONE!");

end


function calculate_loss(w, state, cnnout,data, ids, batchsize, check=false)
	total_loss = 0
	for index in 1:length(data)
		
		cout = convert(KnetArray, Array(Float32, batchsize, cnnout))
		
		for batchno in 1:batchsize
			id = ids[batchno,index]
			if haskey(features, id)
				cout[batchno, :] = features[id]
			else 
				println("No feature found with id: $id")
			end
		end

		#cout = convert(o[:atype], cout)
		batch = data[index]
	
		#To check whether the train_ids are correctly matched to the captions.
		if index%ceil(length(data)/20)==0
			Knet.gc(); gc();
			println(Knet.gpufree(), " GPU memory left")
			if check
				rand_no = rand(1:batchsize)
				id_print = ids[rand_no,index]
				println("ID: $(id_print)")
				for word in batch
					print(index2word[word[rand_no]], " ")
				end	
				println()
			end
		end
		
		total_loss += loss(w,copy(state),cout,batch)
	end

	return total_loss/length(data)
	
end


function loss(w,s,cout,sequence,range=1:length(sequence)-1; pdrop=0.0)
    total = 0.0; count = 0
	
	rnn(w,s,cout*w[1];pdrop=pdrop);
	batchsize = length(sequence[1])
	vocabsize = length(word2index)
	
	prediction = rnn(w,s,w[end][ones(batchsize),:];pdrop=pdrop)
	
	golds = sequence[1]
	#index = similar(golds)
	ynorm = logp(prediction,2)
	@inbounds for i=1:batchsize
		#index[i] = i + (golds[i]-1)*batchsize
		total += ynorm[i,golds[i]]
	end
	
	#total += sum(logp(prediction, 2)[index])
	count += batchsize
	
    for t in range
		prediction = rnn(w,s,w[end][sequence[t],:]; pdrop = pdrop)
		ynorm = logp(prediction,2)
		
		golds = sequence[t+1]

		@inbounds for i=1:batchsize
			#index[i] = i + (golds[i]-1)*batchsize
			total += ynorm[i,golds[i]]
		end
		
		#total += sum(logp(prediction, 2)[index])
		count += batchsize
    end
	
	prediction = rnn(w,s,w[end][sequence[end],:]; pdrop=pdrop)
	
	total += sum(logp(prediction, 2)[:,2])
	count += batchsize
	
    return -total / count
end

lossgradient = grad(loss);

function train(w, state, valstate, train_ids, train_data, valid_ids, valid_data, word2index, o, opts; cnnout = 4096)
	
	index2word = Array(String, length(word2index));
    	for (k,v) in word2index; index2word[v] = k; end
	
	info("Calculating validation loss...")
	val_loss = calculate_loss(w, copy(valstate), cnnout, valid_data, valid_ids, 1, o[:check])		
	best_val_loss = val_loss
	
	if best_val_loss<3.0673
		println("Saving...")
		save(o[:savefile], "model", w)
	end	
	println("Avg validation loss for initial model: ", val_loss)	
	
	info("Calculating training loss...")
	train_loss = calculate_loss(w, copy(state), cnnout, train_data, train_ids, o[:batchsize], o[:check])
	best_train_loss = train_loss

	println("Avg training loss for initial model: ", train_loss)
	loss_string =  "Epoch 0,  Train loss : $train_loss   Validation loss : $val_loss"
	Knet.knetgc(); gc();
	
	for epoch = 1:o[:epochs]

		println("Starting ", epoch,". epoch...")
		for index in 1:length(train_data)
			if index%200==1 && index != 1
				new_state = initstate(o[:atype],o[:hidden],1)
				id_train = train_ids[rand(1:end)]
				println("Generating caption for $(id_train) in train: ")

				generate_caption(id_train, w, copy(new_state), word2index, o[:generate])
				if index%1000==1
					println("Generating with beamsearch with beamsize 1")
					generate_with_beam_search(id_train, w, copy(new_state), word2index, 40, 1)
					println("Generating with beamsearch with beamsize 5")
					generate_with_beam_search(id_train, w, copy(new_state), word2index, 40, 5)	
					println("Generating with beamsearch with beamsize 10")
					generate_with_beam_search(id_train, w, copy(new_state), word2index, 40, 10)	
					println("Generating with beamsearch with beamsize 20")
					generate_with_beam_search(id_train, w, copy(new_state), word2index, 40, 20)	
				end
				
				id_val = valid_ids[rand(1:end)]
				println("Generating caption for $(id_val) in validation: ")
				
				generate_caption(id_val, w, copy(new_state), word2index, o[:generate])
				if index%1000==1
					println("Generating with beamsearch with beamsize 1")
					generate_with_beam_search(id_val, w, copy(new_state), word2index, 40, 1)
					println("Generating with beamsearch with beamsize 5")
					generate_with_beam_search(id_val, w, copy(new_state), word2index, 40, 5)	
					println("Generating with beamsearch with beamsize 10")
					generate_with_beam_search(id_val, w, copy(new_state), word2index, 40, 10)	
					println("Generating with beamsearch with beamsize 20")
					generate_with_beam_search(id_val, w, copy(new_state), word2index, 40, 20)	
				end

				Knet.knetgc(); gc();
			end
			
			#length(train_data[index])>20 && continue;
			#println("Length of the sentence is: ", length(train_data[index]))
			
			cout = Array(Float32, o[:batchsize], cnnout)
			
			for batchno in 1:o[:batchsize]
				id = train_ids[batchno,index]
				cout[batchno, :] = features[id]
			end
			
			cout = convert(o[:atype], cout)
			sentence = train_data[index] #sentence = map(k->convert(o[:atype], k),train_data[index])

			gloss = lossgradient(w,copy(state),cout,sentence; pdrop = o[:dropout])

			update!(w, gloss, opts)

			if index%10 == 1
				#To check whether the train_ids are correctly matched to the captions.
				if o[:check]
					rand_no = rand(1:o[:batchsize])
					id_print = train_ids[rand_no,index]
					println("ID: $(id_print)")
					for word in sentence
						print(index2word[word[rand_no]], " ")
					end	
					println()
				end
				
				@printf("%d is trained %0.3f%% of epoch is completed.\n",index, index/length(train_data)*100)
				println("$(Knet.gpufree()) GPU memory left");
				println("loss in this sentence is: ", loss(w,copy(state),cout,sentence))
			end
		end
		
		println(epoch,". epoch is finished.")
		
		println("Saving just in case...")
		save("./model/lol.jld","model", w, "vocab", word2index, "epochs", epoch)
		
		Knet.gc(); gc();
		println("Calculating training loss...")
		train_loss = calculate_loss(w, copy(state), cnnout, train_data, train_ids, o[:batchsize])
		Knet.gc(); gc();
		println("Epoch $(epoch), average training loss is: ", train_loss)
		
		println("Best training loss was: ", best_train_loss)
		
		println("Calculating validation loss...")
		val_loss = calculate_loss(w, copy(valstate), cnnout, valid_data, valid_ids, 1)		
		Knet.gc(); gc();
		
		println("Epoch $(epoch), average validation loss is: ", val_loss)
		
		println("Best validation loss was: ", best_val_loss)
		
		loss_string *= "Epoch epoch :  Train loss: $train_loss   Validation loss: $val_loss\n"
		
		file = open("model_4_loss.txt", "w")
		write(file, loss_string)
		close(file)
		
		if val_loss<best_val_loss
			println("Saving model...")
			save(o[:savefile],"model", w, "vocab", word2index, "epochs", epoch)
			best_val_loss = val_loss
		end
		
		if train_loss<best_train_loss
			best_train_loss = train_loss
		end
		
	end
end


function init_params(model)
	prms = Array(Any, length(model))
	for i in 1:length(model)
		prms[i] = Adam()
	end
	return prms
end


function minibatch(dict, batchsize)
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
		
		sentence = [ zeros(Int, batchsize) for j=1:length(words) ]
		for i in index:index+batchsize-1
			seq = dict[i];
			id = seq[1];
			len = seq[2];
			words = seq[3];
			for k = 1:len
				sentence[k][i-index+1] = get(word2index,words[k],3)
			end
			
			ids[i-index+1,count] = id;
		end
		push!(data, sentence);
		
		count += 1
		index += batchsize
	end
	#shuffle!(data)
	order = randperm(length(data))
	data = data[order];
	ids = ids[:,order];
	info("Minibatch completed with $(count-1) batches of size $batchsize")
	return ids, data
end


function generate_caption(img, w, state, word2index, nwords; normalized=false, file=nothing)
	if typeof(img)==String
		img = processImage(img, averageImage)
		cnnout = convnet(img)
	
		if normalized
			inputRNN = (cnnout./sum(cnnout))*w[1]
		else
			inputRNN = cnnout*w[1]
		end
	else
		!haskey(features,img) && return;
	
		inputRNN = convert(KnetArray{Float32}, features[img])*w[1]
	end
		
	input = rnn(w, state, inputRNN)
	input = getStartWord(1, length(word2index))
	index = 1
	
	for t in 1:nwords
		ypred = rnn(w, state, input*w[end])
		input[index] = 0
        index = sample(exp(logp(ypred)))
		if file!=nothing
			if index==2
				write(file, "\n")
				return;
			end
			if index!=3
				write(file, index2word[index], " ")
			end
		else
			if(index2word[index]=="</s>")
				println();
				return;
			end
			print(index2word[index], " ")
		end
		input[index] = 1
    end
end


function generate_with_beam_search(img, w, state, word2index, nwords, beamsize; normalized = false, file=nothing)

	if typeof(img)==String
		img = processImage(img, averageImage)
		cnnout = convnet(img)
	
		if normalized
			inputRNN = (cnnout./sum(cnnout))*w[1]
		else
			inputRNN = cnnout*w[1]
		end
	else
		!haskey(features,img) && return;
	
		inputRNN = convert(KnetArray{Float32}, features[img])*w[1]
	end
	
	rnn(w, state, inputRNN)
	
	nodes = [([1], 1.0, i) for i in 1:beamsize]
	
	ypred = rnn(w, state, w[end][[1],:])
	ynorm = exp(logp(ypred))
	ynorm = convert(Array{Float32},ynorm);
	ynorm = reshape(ynorm, length(ynorm))
	new_nodes = typeof(nodes)()
	maxindexes = sortperm(ynorm, rev=true)[1:beamsize]
	
	for j in 1:beamsize
		push!(new_nodes, ([nodes[j][1]; maxindexes[j]], nodes[j][2]*ynorm[maxindexes[j]], j))
	end
	nodes = new_nodes
	
	states = [copy(state) for i in 1:beamsize]

	for t in 1:nwords-1
		sum([nodes[i][1][end]==2 for i in 1:beamsize]) == beamsize && break;
		
		state_flags = trues(beamsize)

		new_states = similar(states)
		new_nodes = typeof(nodes)()
		ended_nodes = typeof(nodes)()
		
		for i in 1:beamsize
			if nodes[i][1][end]!=2
				ypred = rnn(w, states[i], w[end][[nodes[i][1][end]],:])
				ynorm = exp(logp(ypred))
				ynorm = convert(Array{Float32},ynorm);
				ynorm = reshape(ynorm, length(ynorm))
				maxindexes = sortperm(ynorm, rev=true)[1:beamsize]
								
				for j in 1:beamsize
					push!(new_nodes, ([nodes[i][1]; maxindexes[j]%length(word2index)], nodes[i][2]*ynorm[maxindexes[j]],i))
				end
			else
				push!(ended_nodes, nodes[i])
			end
		end

		new_nodes = sort([new_nodes;ended_nodes], by=t->t[2], rev=true)[1:beamsize]
				
		for j in 1:beamsize
			old_index = new_nodes[j][3]

			if state_flags[old_index]
				new_states[j] = states[old_index]
				state_flags[old_index] = false
			else
				new_states[j] = copy(states[old_index])
			end
		end
		
		nodes = new_nodes
		states = new_states
		
    end
	
	for i in 2:length(nodes[1][1])
		word = index2word[nodes[1][1][i]]
		if file!=nothing
			if word=="</s>"
				write(file, "\n")
				return;
			end
			if word != "</u>"
				write(file, word, " ")
			end
		else
			if(word=="</s>")
				println();
				return;
			end
			print(word, " ")
		end
	end

	if file!= nothing
		println();
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
	
	cdim = length(b1[1])
    if cdim != 3
        c1 = cat(3,channelview(b1),channelview(b1),channelview(b1))
        c1 = permutedims(c1,[2,1,3])
    else
        c1 = permutedims(channelview(b1), (3,2,1))
    end
	
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


function rnn(w,s,input; start = 1, pdrop=0)
	for i=1:2:length(s)
		input = dropout(input,pdrop)
		(s[i],s[i+1]) = lstm(w[start + i],w[start + i+1],s[i],s[i+1],input)
		input = s[i]
	end
	input = dropout(input,pdrop)
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
