% Example: Save one 3D variable shapedirs to JSON

% Load your data
load('model.mat');  % only load shapedirs

% Put shapedirs into a struct, so you can give it a name in JSON
S.shapedirs = shapedirs;
S.J_regressor = J_regressor;
S.f = f;
S.kintree_table = kintree_table;
S.posedirs = posedirs;
S.v_template = v_template;
S.weights = weights;

% Convert the struct to a JSON string
jsonStr = jsonencode(S);

% Write the JSON string to a file
fid = fopen('model.json', 'w');
fwrite(fid, jsonStr, 'char');
fclose(fid);

disp('Saved shapedirs to shapedirs.json');