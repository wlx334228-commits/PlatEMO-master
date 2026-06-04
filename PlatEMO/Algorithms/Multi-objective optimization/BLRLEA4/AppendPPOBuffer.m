function buffer = AppendPPOBuffer(buffer,newData)
% Append one or more transitions to the rollout buffer.

    fields = fieldnames(buffer);
    for i = 1 : numel(fields)
        name = fields{i};
        if isfield(newData,name) && ~isempty(newData.(name))
            buffer.(name) = [buffer.(name);newData.(name)];
        end
    end
end
