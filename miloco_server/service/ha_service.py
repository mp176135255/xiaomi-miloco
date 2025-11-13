# Copyright (C) 2025 Xiaomi Corporation
# This software may be used and distributed according to the terms of the Xiaomi Miloco License Agreement.

"""
Home Assistant service module
"""

import logging
from typing import List, Optional

from miloco_server.mcp.mcp_client_manager import MCPClientManager
from miloco_server.middleware.exceptions import (
    HaServiceException,
    ValidationException,
    BusinessException
)
from miloco_server.proxy.ha_proxy import HAProxy
from miloco_server.schema.miot_schema import HAConfig
from miloco_server.schema.trigger_schema import Action
from miloco_server.utils.default_action import DefaultPresetActionManager

from miot.types import HAAutomationInfo

logger = logging.getLogger(__name__)


class HaService:
    """Home Assistant service class"""

    def __init__(
        self,
        ha_proxy: HAProxy,
        mcp_client_manager: MCPClientManager,
        default_preset_action_manager: Optional[DefaultPresetActionManager] = None
    ):
        self._ha_proxy = ha_proxy
        self._mcp_client_manager = mcp_client_manager
        self._default_preset_action_manager = default_preset_action_manager

    @property
    def ha_client(self) -> Optional[object]:
        """Get the HAHttpClient instance."""
        return self._ha_proxy.ha_client

    async def refresh_ha_automations(self):
        """
        Refresh Home Assistant automation information
        """
        try:
            await self._ha_proxy.refresh_ha_automations()
        except Exception as e:
            logger.error("Failed to refresh Home Assistant automations: %s", e)
            raise HaServiceException(f"Failed to refresh Home Assistant automations: {str(e)}") from e

    async def set_ha_config(self, ha_config: HAConfig):
        try:
            if not ha_config.base_url or not ha_config.base_url.strip():
                raise ValidationException("Home Assistant base URL cannot be empty")
            if not ha_config.token or not ha_config.token.strip():
                raise ValidationException("Home Assistant access token cannot be empty")

            await self._ha_proxy.set_ha_config(ha_config.base_url,
                                                    ha_config.token.strip())

            await self._mcp_client_manager.init_ha_automations()
            logger.info("Home Assistant configuration saved successfully: base_url=%s", ha_config.base_url)

        except ValidationException:
            raise
        except Exception as e:
            logger.error("Exception occurred while saving Home Assistant configuration: %s", e)
            raise BusinessException(f"Failed to save Home Assistant configuration: {str(e)}") from e

    async def get_ha_config(self) -> HAConfig | None:
        try:
            ha_config = self._ha_proxy.get_ha_config()
            if not ha_config:
                logger.warning("Home Assistant configuration not set")
            return ha_config
        except Exception as e:
            logger.error("Exception occurred while getting Home Assistant configuration: %s", e)
            raise HaServiceException(f"Failed to get Home Assistant configuration: {str(e)}") from e

    async def get_ha_automations(self) -> list[HAAutomationInfo]:
        try:
            automations = await self._ha_proxy.get_automations()
            if automations is None:
                logger.warning("Failed to get Home Assistant automation list")
                raise HaServiceException("Failed to get Home Assistant automation list")
            logger.info(
                "Successfully retrieved Home Assistant automation list - count: %d", len(automations.values()))
            return list(automations.values())

        except Exception as e:
            logger.error("Failed to get Home Assistant automation list: %s", e)
            raise HaServiceException(
                f"Failed to get Home Assistant automation list: {str(e)}") from e

    async def get_ha_automation_actions(self) -> List[Action]:
        """
        Get Home Assistant automation action list

        Returns:
            List[Action]: Home Assistant automation action list

        Raises:
            HaServiceException: When getting automation actions fails
        """
        try:
            if not self._default_preset_action_manager:
                logger.error("DefaultPresetActionManager not initialized")
                raise HaServiceException("DefaultPresetActionManager not initialized")

            actions = await self._default_preset_action_manager.get_ha_automation_actions()

            return list(actions.values())
        except Exception as e:
            logger.error("Failed to get Home Assistant automation action list: %s", e)
            raise HaServiceException(f"Failed to get Home Assistant automation action list: {str(e)}") from e
