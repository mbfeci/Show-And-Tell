for p in ("MAT", "JLD", "JSON")
    Pkg.installed(p) == nothing && Pkg.add(p)
end

using MAT, JLD, JSON

dataset = open("./data/Flickr30k/VGG/dataset.json")
images = JSON.parse(dataset)["images"]

feats = matread("./data/Flickr30k/VGG/vgg_feats.mat")["feats"]

index = 0
for image in images
	feat = feats[:,image["imgid"]+1]'
	img_id = image["filename"][1:rsearch(image["filename"],'.')-1]
	save("./data/Flickr30k/VGG/features/$(img_id).jld", "feature", feat);
	index += 1
	index%100==0 && println(index, " number of features have been saved successfully.")
end
