# Copyright (C) 2025 Xiaomi Corporation
# This software may be used and distributed according to the terms of the Xiaomi Miloco License Agreement.

"""
Device chooser utility for device selection.
Currently supports camera selection with location-based filtering.
"""

import json
import logging
from typing import List, Optional

from miloco_server.utils.llm_utils.base_llm_util import BaseLLMUtil

from miloco_server.schema.miot_schema import CameraInfo
from miloco_server.utils.normal_util import extract_json_from_content


logger = logging.getLogger(__name__)


class DeviceChooser(BaseLLMUtil):
    """For device selection, currently only supports camera selection, singleton implementation"""

    def __init__(self,
                 request_id: str,
                 location: Optional[str] = None,
                 choose_camera_device_ids: Optional[List[str]] = None):
        super().__init__(request_id=request_id, tools_meta=None)
        self._location = location
        self._choose_camera_device_ids = choose_camera_device_ids
        self._choosed_cameras = []
        self._all_cameras = []

    def _get_system_prompt(self) -> str:
        """Get system prompt"""
        return """
        Device selector, select devices based on location.
        Next I will give you a set of device information and the location the user wants. You need to select devices based on location information and return the device did.
        You can only return in JSON format, JSON format is:
        {
            "device_ids": ["did1", "did2", "did3"]
        }
        """

    def _init_conversation(self) -> None:
        self._chat_history.add_content("system", self._get_system_prompt())
        self._chat_history.add_content(
            "user", f"User desired location: {self._location}, "
            f"Device information: {json.dumps([camera.model_dump() for camera in self._all_cameras])}"
        )

    async def _choose_camera(
            self) -> tuple[List[CameraInfo], List[CameraInfo]]:
        """Choose camera"""
        try:
            self._all_cameras = await self._manager.miot_service.get_miot_camera_list(
            )
            if not self._all_cameras:
                return [], []

            if self._choose_camera_device_ids:
                self._choosed_cameras = [
                    c for c in self._all_cameras
                    if c.did in self._choose_camera_device_ids
                ]
                return self._choosed_cameras, self._all_cameras

            if not self._location:
                return self._all_cameras, self._all_cameras

            self._init_conversation()
            content, _, _ = await self._call_llm()

            if not content:
                return [], self._all_cameras

            json_content = extract_json_from_content(content)
            if not json_content:
                raise ValueError(f"No JSON in LLM response: {content}")

            device_ids = json.loads(json_content).get("device_ids", [])
            if not device_ids:
                raise ValueError(
                    f"No device_ids in LLM response: {json_content}")

            return [c for c in self._all_cameras if c.did in device_ids], self._all_cameras
        except Exception as e: # pylint: disable=broad-exception-caught
            logger.error("[%s] Error occurred during device chooser: %s",
                         self._request_id,
                         str(e),
                         exc_info=True)
            return [], self._all_cameras

    def _format_camera_info(self, cameras: List[CameraInfo]) -> List[str]:
        """Format camera information for logging"""
        return [
            f"{c.did}_{c.name}_{c.home_name}_{c.room_name}" for c in cameras
        ]

    async def run(self) -> tuple[List[CameraInfo], List[CameraInfo]]:
        choosed_camera_list, all_camera_list = await self._choose_camera()
        logger.info("Choosed camera did with name: %s", self._format_camera_info(choosed_camera_list))
        logger.info("All camera did with name: %s", self._format_camera_info(all_camera_list))
        return choosed_camera_list, all_camera_list
