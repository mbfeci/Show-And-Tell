



function parse_files(caption_files)
	dicts = Array{Array{Tuple{Int64, Tuple{Int64, Array{String, 1}}},1},1}();
	
	for i=1:length(caption_files)
		file = open(caption_files[i])
		push!(dicts, parse_flickr30k(readlines(file)))
	end
	
	vocab = createVocabulary(dicts)
	return dicts, vocab
end

function parse_flickr30k(lines)
	dict = Array{Tuple{Int64, Tuple{Int64, Array{String, 1}}},1}();  # (id, (#no, Sentence))
	for i = 1 : length(lines)
		tokens = map(lowercase, split(lines[i],['.','#','\t',' ', '\n']))
		filter!(t->length(t)>0,tokens);
		push!(dict, (parse(Int64, tokens[1]), (parse(Int64, tokens[3]), tokens[4:end])))  
	end
	sort!(dict; by = t->t[1])
	return dict
end

function parse_flickr8k()

end

function parse_mscoco()

end

function createVocabulary(dicts)
    vocab = Dict{String,Int}()
	get!(vocab, "<s>", 1)       #Special start token
	get!(vocab, "</s>", 2)      #Special end token
	for dict in dicts
		for element in dict
			for word in element[2][2]
				get!(vocab, word, length(vocab)+1)
			end
		end
	end
    return vocab
end

