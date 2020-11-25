package com.julanling.pmml;

import com.alibaba.fastjson.JSONObject;
import org.dmg.pmml.FieldName;
import org.jpmml.evaluator.EvaluatorUtil;
import org.jpmml.evaluator.FieldValue;
import org.jpmml.evaluator.LoadingModelEvaluatorBuilder;
import org.jpmml.evaluator.ModelEvaluator;
import org.xml.sax.SAXException;

import javax.xml.bind.JAXBException;
import java.io.File;
import java.io.IOException;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Random;

/**
 * @author admin
 */
public class ClassificationModel {
    public static ModelEvaluator<?> evaluator;

    static {
        try {
            evaluator = new LoadingModelEvaluatorBuilder()
                    .load(new File("E:\\pycharmProject\\cat\\pmml\\titanic.pmml"))
                    .build();
            evaluator.verify();
            System.out.println("load model success");
        } catch (SAXException | JAXBException | IOException e) {
            System.out.println("load model fail");
            e.printStackTrace();
        }
    }

    public static String getEvaluate(Map<FieldName, FieldValue> arguments) {
        Map<FieldName, ?> evaluate = evaluator.evaluate(arguments);
        Map<String, ?> resultRecord = EvaluatorUtil.decodeAll(evaluate);
        return JSONObject.toJSONString(resultRecord);
    }

    public static Map<FieldName, FieldValue> getArgs() {
        Map<FieldName, FieldValue> arguments = new LinkedHashMap<>();
        Random random = new Random();
        evaluator.getInputFields().forEach(inputField -> {
            FieldName inputFieldName = inputField.getName();
            int nextInt = random.nextInt(1000);
            FieldValue fieldValue = inputField.prepare(nextInt);
            arguments.put(inputFieldName, fieldValue);
        });
        return arguments;
    }

    public static void main(String[] args) {
        String evaluate = getEvaluate(getArgs());
        System.out.println(evaluate);
    }

}
