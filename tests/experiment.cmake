# Exercise the same train and predict commands used in the README.
cmake_minimum_required(VERSION 3.16)
string(RANDOM LENGTH 14 ALPHABET 0123456789abcdef suffix)
set(root "${CMAKE_CURRENT_BINARY_DIR}/spiralnet-test-${suffix}")
if(EXISTS "${root}")
    message(FATAL_ERROR "Testverzeichnis existiert bereits")
endif()
file(MAKE_DIRECTORY "${root}")

foreach(run IN ITEMS a b)
    execute_process(COMMAND "${PROGRAM}" train --out "${root}/${run}"
                    --epochs 3 --width 5 --seed 0
                    RESULT_VARIABLE status)
    if(NOT status EQUAL 0)
        message(FATAL_ERROR "Trainingslauf ${run} fehlgeschlagen")
    endif()
endforeach()

# Equal seeds on the same build must reproduce the split, all grid frames,
# the selected checkpoint, and all evaluation outputs byte for byte.
foreach(name IN ITEMS model.nn points.csv history.csv predictions.csv summary.csv surface.csv)
    file(SHA256 "${root}/a/${name}" first)
    file(SHA256 "${root}/b/${name}" second)
    if(NOT first STREQUAL second)
        message(FATAL_ERROR "Lauf nicht reproduzierbar: ${name}")
    endif()
endforeach()

file(STRINGS "${root}/a/history.csv" history)
list(REMOVE_AT history 0)
set(minimum 1e300)
foreach(row IN LISTS history)
    string(REPLACE "," ";" metrics "${row}")
    list(GET metrics 2 validation_loss)
    if(validation_loss LESS minimum)
        set(minimum "${validation_loss}")
        list(GET metrics 0 expected_epoch)
    endif()
endforeach()
file(STRINGS "${root}/a/summary.csv" summary)
list(GET summary 1 data)
string(REPLACE "," ";" data "${data}")
list(GET data 3 selected_epoch)
if(NOT selected_epoch STREQUAL expected_epoch)
    message(FATAL_ERROR "Modellauswahl weicht vom Minimum des Validierungsverlusts ab")
endif()

file(STRINGS "${root}/a/predictions.csv" rows)
list(LENGTH rows n)
if(NOT n EQUAL 301)
    message(FATAL_ERROR "Testauswertung muss genau 300 Vorhersagen enthalten")
endif()
list(GET rows 1 record)
string(REPLACE "," ";" fields "${record}")
list(GET fields 0 x1)
list(GET fields 1 x2)
list(REMOVE_AT fields 2)  # Drop the known label from the evaluation CSV.
list(JOIN fields "," expected)
execute_process(COMMAND "${PROGRAM}" predict "${root}/a/model.nn" "${x1}" "${x2}"
                OUTPUT_VARIABLE output RESULT_VARIABLE status OUTPUT_STRIP_TRAILING_WHITESPACE)
string(REPLACE "\r" "" output "${output}")
if(NOT status EQUAL 0 OR NOT output STREQUAL "x1,x2,pred,p0,p1,p2\n${expected}")
    message(FATAL_ERROR "Inferenz aus dem gespeicherten Modell weicht von der Testauswertung ab")
endif()

configure_file("${root}/a/model.nn" "${root}/invalid.nn" COPYONLY)
file(APPEND "${root}/invalid.nn" "x")
execute_process(COMMAND "${PROGRAM}" predict "${root}/invalid.nn" 0 0
                RESULT_VARIABLE status ERROR_QUIET)
if(status EQUAL 0)
    message(FATAL_ERROR "Beschaedigtes Modell wurde faelschlich akzeptiert")
endif()

file(REMOVE_RECURSE "${root}")
