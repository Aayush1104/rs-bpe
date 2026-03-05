use bpe_openai::deepseek_v32::{apply_chat_template, Message, ThinkingMode};

#[test]
fn test_apply_chat_template_matches_official_test_input_output() {
    let input_raw = include_str!("fixtures/deepseek_v32/test_input.json");
    let expected_raw = include_str!("fixtures/deepseek_v32/test_output.txt");

    let test_cases: Vec<Vec<Message>> =
        serde_json::from_str(input_raw).expect("fixture test_input.json should be valid");
    let expected_lines: Vec<&str> = expected_raw
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .collect();

    assert_eq!(
        test_cases.len(),
        expected_lines.len(),
        "fixture size mismatch between input and output"
    );

    for (index, (messages, expected_line)) in
        test_cases.iter().zip(expected_lines.iter()).enumerate()
    {
        let prompt = apply_chat_template(messages, ThinkingMode::Chat, None, true, true)
            .expect("apply_chat_template should succeed");
        let prompt_json = serde_json::to_string(&prompt)
            .expect("serializing prompt as json string should succeed");

        assert_eq!(
            prompt_json, *expected_line,
            "prompt mismatch at case #{index}"
        );
    }
}
