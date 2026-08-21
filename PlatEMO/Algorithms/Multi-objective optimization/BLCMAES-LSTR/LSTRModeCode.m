function code = LSTRModeCode(mode)
% Convert the LSTR stage name to a compact archive code.

    switch lower(mode)
        case 'feasibility'
            code = 1;
        case 'objective'
            code = 3;
        otherwise
            code = 2;
    end
end
