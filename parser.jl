function parse_file(caption_file)
	dict = Array{Tuple{Int64, Int64, Array{String, 1}},1}();
	
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
		push!(dict, (parse(Int64, tokens[1]), length(tokens)-3, map(t->strip(t,[' ', '.', ',','#', '\'', ')', '(', '!', '/', '?', '\t', '`']),tokens[4:end]))) 
	end
	sort!(dict; by = t->t[2])
	return dict
end

function parse_flickr8k()

end

function parse_mscoco()

end

function createVocabulary(dict)
    vocab = Dict{String,Int}()
	get!(vocab, "<s>", 1)       #Special start token
	get!(vocab, "</s>", 2)      #Special end token
	for element in dict
		for word in element[3]
			get!(vocab, word, length(vocab)+1)
		end
	end
    return vocab
end

