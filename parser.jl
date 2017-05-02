
function parse_file(caption_file)
	dict = Array{Tuple{Int64, Int64, Array{String, 1}},1}();
	
#TODO: Understand if the file is MSCOCO or Flickr

	#dict = parse_mscoco(caption_file)
	
	file = open(caption_file)
	dict = parse_flickr30k(readlines(file))
	
	vocab = createVocabulary(dict)
	
	return dict, vocab
end

function parse_flickr30k(lines)
	dict = Array{Tuple{Int64, Int64, Array{String, 1}},1}();  # (id, (#no, Sentence))
	
	for i = 1 : length(lines)
		tokens = map(lowercase, split(lines[i],['.','#','\t',' ', '\n']))
		filter!(t->length(t)>0, tokens);
		push!(dict, (parse(Int64, tokens[1]), length(tokens)-3, map(t->strip(t,[' ', '.', ',','#', '\'', ')', '(', '!', '/', '?', '\t', '`', '"']),tokens[4:end]))) 
	end
	sort!(dict; by = t->t[2])
	return dict
end

function parse_flickr8k()

end

function parse_mscoco(fileDir)
	dict = Array{Tuple{Int64, Int64, Array{String, 1}},1}();
	
	data = JSON.parsefile(fileDir)
	annotations = data["annotations"]
	
	for i = 1 : length(annotations)
		tokens = map(lowercase, split(annotations[i]["caption"],['.','#','\t',' ', '\n']))
		filter!(t->length(t)>0, tokens);
		push!(dict, (annotations[i]["image_id"], length(tokens), map(t->strip(t,[' ', '.', ',','#', '\'', ')', '(', '!', '/', '?', '\t', '`', '"']),tokens))) 
	end

	sort!(dict; by = t->t[2])
	return dict
end

function splitdata(dict, validsize, testsize)
	srand(1)
	train_dict = Array{Tuple{Int64, Int64, Array{String, 1}},1}();
	valid_dict = Array{Tuple{Int64, Int64, Array{String, 1}},1}();
	test_dict  = Array{Tuple{Int64, Int64, Array{String, 1}},1}();

	ids = [element[1] for element in dict]
	ids = union(ids)
	
	matchdict = Dict{Int64, Int}()
	
	shuffle!(ids)
	
	for i=1:validsize
		get!(matchdict, ids[i], 1) #1 for validation set
	end
	
	for i=validsize+1:validsize+testsize
		get!(matchdict, ids[i], 2) #2 for test set
	end
	
	#=
	valid_ids = ids[1:validsize]

	test_ids = ids[validsize+1:validsize+testsize]
	=#
	
	for element in dict
		id = element[1]
		
		group = get(matchdict, id, 0) #0-> training 1-> validation 2-> test
		
		group==0 && push!(train_dict, element)
		group==1 && push!(valid_dict, element)
		group==2 && push!(test_dict, element)
	end
	
	srand();
	return train_dict, valid_dict, test_dict
end


function createVocabulary(dict)
    vocab = Dict{String,Int}()
	counter = Dict{String, Int}()
	get!(vocab, "<s>", 1)       #Special start token
	get!(vocab, "</s>", 2)      #Special end token
	get!(vocab, "</u>", 3)		#Special unknown token
	
	for element in dict
		for word in element[3]
			counter[word] = get(counter, word, 0)+1
		end
	end
	
	for word in keys(counter)
		counter[word] > 4 && get!(vocab, word, length(vocab)+1)
	end
    return vocab
end

