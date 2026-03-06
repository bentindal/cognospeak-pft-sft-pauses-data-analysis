import csv

mammals = [
    "aardvark", "alpaca", "alpacas", "ant eater", "anteater", "antelope",
    "antelopes", "anterlope", "ape", "armadillo", "ass", "baboon", "baboons",
    "badger", "bat", "bear", "bears", "beaver", "beavers", "big cats",
    "big dogs", "bigcats?", "bison", "black bear", "blue whale", "bobcat",
    "bonobo", "brown bear", "brown bears", "buffalo", "bull", "bulls", "calf",
    "calves", "camel", "camels", "camle", "cariboo", "cat", "cats", "cheetah",
    "cheetahs", "chihuahua", "chimpanzee", "chimpanzees", "civet", "cougar",
    "cougars", "cow", "cows", "coyote", "dear", "deer", "deers", "dingo",
    "dog", "dogs", "dolphin", "dolphins", "donkey", "donkeys", "dormouse",
    "echidnea", "eland", "elephant", "elephants", "elk", "elks", "ferret",
    "field mice", "foal", "fox", "foxes", "gazelle", "genet", "gerbil",
    "gerbils", "giraffe", "giraffes", "gnus", "goat", "goat?", "goats",
    "gorilla", "gorillas", "grey squirrel", "grizzly bear", "grizzly bears",
    "guinea pig", "guinea pigs", "hamster", "hamsters", "hare", "hares",
    "harvest mouse", "hedgehog", "hedgehogs", "hippo", "hippopotamus",
    "hippos", "hippotamus", "horse", "horses", "hound", "human", "hyena",
    "hyenas", "ibex", "jackal", "jaguar", "kangaroo", "kangaroos", "kitten",
    "kittens", "koala", "koala bear", "koala bears", "lamb", "lambs",
    "lap dog", "lemur", "leopard", "leopards", "lima", "limas", "lion",
    "lions", "little cats", "little dogs", "little mice", "llama", "llamas",
    "loris?", "lynx", "lynx?", "marmoset", "marmoset?", "meerkat", "meerkats",
    "mice", "minks", "mole", "mongoose", "monkey", "monkeys", "moose",
    "mouse", "mule", "mules", "naked pig", "ocelot", "opossums", "orange cat",
    "orangutan", "oranutang", "otter", "otters", "ox", "oxen", "panda",
    "pandas", "pangolin", "pangolins", "panther", "panthers", "pig", "pigs",
    "pigmy shrew", "platapus", "platypus", "polar bear", "polar bears",
    "polecat", "ponies", "pony", "poochie", "porcupine", "porcupines",
    "porpoise", "possum", "prairie dog", "puma", "pumas", "puppies", "puppy",
    "pussycats", "rabbit", "rabbits", "raccoons", "racing horses", "racoon",
    "rat", "rats", "red bear", "red squirrel", "reindeer", "reinders",
    "rhinceros", "rhino", "rhinoceros", "rhinos", "right whale", "rodents",
    "sea lion", "seal", "sealions", "sheep", "shetland pony", "shrewd",
    "skunks", "sloth", "snow bear", "snow leopard", "spaniels", "springbok",
    "squirrel", "squirrels", "stag", "stags", "stoat", "stort", "stote",
    "tasmanian devil", "tiger", "tiger cat", "tigers", "vixen", "wallabee",
    "wallabees", "wallabies", "wallaby", "walrus", "warthog", "water rats",
    "weasel", "weasels", "wheasil", "whale", "whales", "white bear",
    "wildebeest", "wilderbeast", "wolf", "wolves", "wombat", "wombats",
    "zebra", "zebras",
]

birds = [
    "bantams", "bird", "birds", "birds?", "black bird", "blackbird",
    "blackbirds", "blue tit", "blue tits", "bluebird", "bluejay",
    "budgerigar", "budgie", "buzzard", "cardinals", "chaffensures",
    "chicken", "chickens", "cock", "cockerels", "col tits", "condor",
    "cormorant", "crow", "crows", "duck", "ducks", "eagle", "eagles",
    "egret", "emu", "emus", "flamingo", "flamingos", "geese", "goose",
    "great tits", "harrier", "harrier pigeon", "hawk", "hen", "hens",
    "hummingbird", "kingfisher", "kiwi", "montail tits", "nuthatch",
    "osprey", "ostrich", "owl", "parakeet", "parrot", "parrots", "partridge",
    "peacock", "penguin", "pheasant", "pheasants", "pigeon", "pigeons",
    "puffin", "raven", "robin", "robins", "sparrow", "sparrows", "swan",
    "toucans", "turkey", "turkeys", "umbrellabird", "woodpecker",
]

reptiles = [
    "adder", "alligator", "alligators", "anaconda", "boa constrictor",
    "crocodile", "crocodiles", "dinosaurs", "gila monster", "iguana",
    "iguanas", "lizard", "lizards", "pterodactyls", "python", "serpent",
    "snake", "snakes", "tortoise", "tortoises", "turtle", "turtles",
]

amphibians = [
    "frog",
]

fish = [
    "angelfish", "chub", "cod", "eels", "fish", "goldfish", "herring",
    "leopard fish", "mackerel", "manta ray", "perch", "plaice", "roach",
    "salmon", "sardine", "shark", "sharks", "sprat", "tiger fish", "tol fish",
    "trout", "tuna", "turbat",
]

insects = [
    "ant", "ants", "bees", "beetle", "black ants", "bugs", "butterfly",
    "caterpillar", "flea", "flies", "greenfly", "insects", "ladybird", "moth",
    "red ants", "wasp", "wasps", "weetle",
]

arachnids = [
    "scorpion", "spiders",
]

invertebrates = [
    "jellyfish", "octopus", "sea urchin", "shellfish", "slug", "snail",
    "snails", "starfish", "woodlice", "worm", "worms",
]

mythical = [
    "unicorn",
]

unknown = [
    "?", "ball", "field ?", "flan", "holibert", "jack", "snow", "unknown?",
]

categories = {}
for label, animals in [
    ("Mammal", mammals),
    ("Bird", birds),
    ("Reptile", reptiles),
    ("Amphibian", amphibians),
    ("Fish", fish),
    ("Insect", insects),
    ("Arachnid", arachnids),
    ("Invertebrate", invertebrates),
    ("Mythical", mythical),
    ("Unknown/Other", unknown),
]:
    for a in animals:
        categories[a] = label

with open("unique_sft_answers.csv", newline="") as infile, \
     open("categories.csv", "w", newline="") as outfile:
    reader = csv.DictReader(infile)
    writer = csv.writer(outfile)
    writer.writerow(["answer", "category"])
    missing = []
    for row in reader:
        answer = row["answer"].strip()
        if not answer:
            continue
        cat = categories.get(answer)
        if cat is None:
            missing.append(answer)
            cat = "Unknown/Other"
        writer.writerow([answer, cat])

if missing:
    print(f"WARNING: {len(missing)} entries not in dictionary: {missing}")
else:
    print(f"Done. All entries categorized. Wrote categories.csv")
