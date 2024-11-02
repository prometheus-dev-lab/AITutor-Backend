import unittest

from AITutor_Backend.src.PromptUtils.prompt_utils import Conversation, Message

# Assuming your code is defined here as Message and Conversation


class TestMessage(unittest.TestCase):

    def test_to_dict(self):
        message = Message("user", "Hello, AI")
        self.assertEqual(message.to_dict(), {"role": "user", "content": "Hello, AI"})

    def test_copy(self):
        original = Message("admin", "Check this out")
        copied = original.copy()
        self.assertNotEqual(id(original), id(copied))
        self.assertEqual(original.to_dict(), copied.to_dict())


class TestConversation(unittest.TestCase):

    def setUp(self):
        self.messages = [
            Message("user", "Hi"),
            Message("system", "Hello, how can I assist?"),
        ]
        self.conversation = Conversation()
        for msg in self.messages:
            self.conversation.append_message(msg)

    def test_append_message_by_instance(self):
        new_msg = Message("user", "Another message")
        self.conversation.append_message(new_msg)
        self.assertIn(new_msg, self.conversation.messages)

    def test_append_message_by_params(self):
        self.conversation.append_message(Message("user", "A new message"))
        self.assertEqual(self.conversation.messages[-1].role, "user")
        self.assertEqual(self.conversation.messages[-1].content, "A new message")

    def test_from_message_list(self):
        new_conversation = Conversation.from_message_list(self.messages)
        self.assertNotEqual(id(new_conversation.messages), id(self.messages))
        for original, copied in zip(self.messages, new_conversation.messages):
            self.assertEqual(original.to_dict(), copied.to_dict())
            self.assertNotEqual(id(original), id(copied))

    def test_to_dict(self):
        messages_dict = self.conversation.to_dict()
        expected = [{"role": msg.role, "content": msg.content} for msg in self.messages]
        self.assertEqual(messages_dict, expected)

    def test_copy(self):
        copied_conversation = self.conversation.copy()
        self.assertNotEqual(
            id(self.conversation.messages), id(copied_conversation.messages)
        )
        self.assertEqual(
            len(self.conversation.messages), len(copied_conversation.messages)
        )
        for original, copied in zip(
            self.conversation.messages, copied_conversation.messages
        ):
            self.assertEqual(original.to_dict(), copied.to_dict())
            self.assertNotEqual(id(original), id(copied))


if __name__ == "__main__":
    unittest.main()
